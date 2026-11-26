/*
 * Copyright 2023 The Kubernetes Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an \"AS IS\" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"errors"
	"fmt"
	"maps"
	"slices"

	"github.com/ROCm/k8s-gpu-dra-driver/pkg/consts"
	klog "k8s.io/klog/v2"

	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdiparser "tags.cncf.io/container-device-interface/pkg/parser"
	cdispec "tags.cncf.io/container-device-interface/specs-go"
)

const (
	cdiVendor = "k8s." + consts.DriverName
	cdiClass  = "gpu"
	cdiKind   = cdiVendor + "/" + cdiClass

	cdiCommonDeviceName = "common"
)

type CDIHandler struct {
	cache    *cdiapi.Cache
	nodeName string
}

func NewCDIHandler(config *Config) (*CDIHandler, error) {
	cache, err := cdiapi.NewCache(
		cdiapi.WithSpecDirs(config.flags.cdiRoot),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create a new CDI cache: %w", err)
	}
	handler := &CDIHandler{
		cache:    cache,
		nodeName: config.flags.nodeName,
	}

	return handler, nil
}

func (cdi *CDIHandler) CreateCommonSpecFile() error {
	spec := &cdispec.Spec{
		Kind: cdiKind,
		Devices: []cdispec.Device{
			{
				Name: cdiCommonDeviceName,
				ContainerEdits: cdispec.ContainerEdits{
					Env: []string{
						fmt.Sprintf("KUBERNETES_NODE_NAME=%s", cdi.nodeName),
						fmt.Sprintf("DRA_RESOURCE_DRIVER_NAME=%s", consts.DriverName),
					},
				},
			},
		},
	}

	minVersion, err := cdiapi.MinimumRequiredVersion(spec)
	if err != nil {
		return fmt.Errorf("failed to get minimum required CDI spec version: %v", err)
	}
	spec.Version = minVersion

	specName, err := cdiapi.GenerateNameForTransientSpec(spec, cdiCommonDeviceName)
	if err != nil {
		return fmt.Errorf("failed to generate Spec name: %w", err)
	}

	return cdi.cache.WriteSpec(spec, specName)
}

// cdiDeviceName builds the CDI device name for one prepared device.
//
// The claim UID and device name alone are not unique: a claim may hold several
// shares of the same synthetic-partition device, which would otherwise produce
// duplicate CDI device names within a single spec. Appending the partition slot
// keeps them distinct and names the physical partition the container receives.
// Devices without a slot (regular GPUs, VFIO) keep the original two-part name.
//
// CreateClaimSpecFile and GetClaimDevices must agree on this name, so both go
// through here.
func cdiDeviceName(claimUID string, device *PreparedDevice) string {
	if device.PartitionSlot == nil {
		return fmt.Sprintf("%s-%s", claimUID, device.DeviceName)
	}
	return fmt.Sprintf("%s-%s-slot%d", claimUID, device.DeviceName, *device.PartitionSlot)
}

func (cdi *CDIHandler) CreateClaimSpecFile(claimUID string, devices PreparedDevices) error {
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClass, claimUID)

	spec := &cdispec.Spec{
		Kind:    cdiKind,
		Devices: []cdispec.Device{},
	}

	for _, device := range devices {
		klog.Infof("Creating CDI spec for device: %+v", device)
		claimEdits := cdiapi.ContainerEdits{}
		claimEdits.Append(device.ContainerEdits)

		cdiDevice := cdispec.Device{
			Name:           cdiDeviceName(claimUID, device),
			ContainerEdits: *claimEdits.ContainerEdits,
		}

		spec.Devices = append(spec.Devices, cdiDevice)
	}

	minVersion, err := cdiapi.MinimumRequiredVersion(spec)
	if err != nil {
		return fmt.Errorf("failed to get minimum required CDI spec version: %v", err)
	}
	spec.Version = minVersion

	return cdi.cache.WriteSpec(spec, specName)
}

// RegenerateClaimSpecFiles rewrites the per-claim CDI spec file of every claim
// recorded in the checkpoint.
//
// The spec files live under --cdi-root (/var/run/cdi in production), which is a
// tmpfs: they are gone after every node reboot. The checkpoint under
// /var/lib/kubelet/plugins is on disk and does survive, and kubelet restores its
// dra_manager_state with the CDI device IDs it already computed, so it never
// calls Prepare for those claims again — the files would stay missing and every
// container of such a claim would fail with "unresolvable CDI devices" until the
// claim is deleted. Everything the spec needs is in the checkpoint (device name,
// partition slot, container edits), so rewriting it at startup is enough.
//
// It runs once per plugin start, never on the Prepare/Unprepare path, and writes
// the same bytes for the same checkpoint, so a restart over a correct CDI root
// changes nothing.
func (cdi *CDIHandler) RegenerateClaimSpecFiles(claims PreparedClaims) error {
	var errs []error
	written := 0
	for _, claimUID := range slices.Sorted(maps.Keys(claims)) {
		devices := claims[claimUID]
		if len(devices) == 0 {
			// Defensive: this binary never records a claim without devices (CDI
			// rejects an empty spec, so the write in Prepare fails before the
			// checkpoint is updated). A nil entry is also what makes Prepare
			// re-drive such a claim, so skipping one loses nothing.
			continue
		}
		// A device the checkpoint records without its container edits would make
		// CreateClaimSpecFile dereference a nil pointer. Name the claim at fault
		// instead of panicking at startup.
		var missing []string
		for _, device := range devices {
			if device == nil {
				missing = append(missing, "<nil>")
			} else if device.ContainerEdits == nil {
				missing = append(missing, device.DeviceName)
			}
		}
		if len(missing) > 0 {
			errs = append(errs, fmt.Errorf("claim %s: checkpoint records no container edits for device(s) %v", claimUID, missing))
			continue
		}
		if err := cdi.CreateClaimSpecFile(claimUID, devices); err != nil {
			errs = append(errs, fmt.Errorf("claim %s: %w", claimUID, err))
			continue
		}
		written++

		// kubelet asks the runtime for exactly the CDI device IDs Prepare
		// returned, not for whatever the file happens to contain. A spec written
		// under different names therefore does not bring the claim back; that can
		// only mean the naming scheme changed under the checkpoint (a driver
		// upgrade), which this function cannot repair — say so.
		for _, device := range devices {
			if len(device.CdiDeviceIds) == 0 {
				continue
			}
			regenerated := cdi.GetClaimDevices(claimUID, []*PreparedDevice{device})
			if !slices.Equal(device.CdiDeviceIds, regenerated) {
				klog.Warningf("claim %s device %s: checkpoint records CDI device IDs %v but the regenerated spec declares %v; kubelet will still request the recorded ones",
					claimUID, device.DeviceName, device.CdiDeviceIds, regenerated)
			}
		}
	}

	if written > 0 {
		klog.Infof("regenerated %d claim CDI spec file(s) from the checkpoint", written)
	}

	return errors.Join(errs...)
}

func (cdi *CDIHandler) DeleteClaimSpecFile(claimUID string) error {
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClass, claimUID)
	return cdi.cache.RemoveSpec(specName)
}

// GetClaimDevices returns the qualified CDI device names for the given prepared
// devices. Names are built with cdiDeviceName so they match exactly what
// CreateClaimSpecFile wrote into the spec.
func (cdi *CDIHandler) GetClaimDevices(claimUID string, devices []*PreparedDevice) []string {
	cdiDevices := []string{
		cdiparser.QualifiedName(cdiVendor, cdiClass, cdiCommonDeviceName),
	}

	for _, device := range devices {
		cdiDevice := cdiparser.QualifiedName(cdiVendor, cdiClass, cdiDeviceName(claimUID, device))
		cdiDevices = append(cdiDevices, cdiDevice)
	}

	return cdiDevices
}
