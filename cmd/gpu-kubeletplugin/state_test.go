/*
 * Copyright 2025 The Kubernetes Authors.
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
	"os"
	"path/filepath"
	"testing"

	"github.com/ROCm/k8s-gpu-dra-driver/pkg/consts"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"

	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdispec "tags.cncf.io/container-device-interface/specs-go"
)

func TestRestoreFromVfio(t *testing.T) {
	original := &AmdGpuInfo{PCIAddress: "0000:0d:00.0", cardIndex: 0, renderIndex: 128}
	state := &DeviceState{
		allocatable: AllocatableDevices{
			"gpu-0-128": {Vfio: &AmdGpuVFIOInfo{PCIAddress: "0000:0d:00.0"}},
		},
		claimVfioConversions: map[string]*AmdGpuInfo{
			"gpu-0-128": original,
		},
	}

	state.restoreFromVfio("gpu-0-128")

	allocDev := state.allocatable["gpu-0-128"]
	assert.NotNil(t, allocDev.AmdGpu, "AmdGpu should be restored")
	assert.Nil(t, allocDev.Vfio, "Vfio should be cleared")
	assert.Equal(t, "0000:0d:00.0", allocDev.AmdGpu.PCIAddress)
	assert.Equal(t, 0, allocDev.AmdGpu.cardIndex)
	assert.Equal(t, 128, allocDev.AmdGpu.renderIndex)
	assert.Equal(t, consts.AmdGpuDeviceType, allocDev.Type())
	_, inMap := state.claimVfioConversions["gpu-0-128"]
	assert.False(t, inMap, "device should be removed from claimVfioConversions")
}

func TestRestoreFromVfio_NoConversion(t *testing.T) {
	state := &DeviceState{
		allocatable: AllocatableDevices{
			"gpu-vfio-0": {Vfio: &AmdGpuVFIOInfo{PCIAddress: "0000:0d:00.0"}},
		},
		claimVfioConversions: map[string]*AmdGpuInfo{},
	}

	state.restoreFromVfio("gpu-vfio-0")

	allocDev := state.allocatable["gpu-vfio-0"]
	assert.NotNil(t, allocDev.Vfio, "pre-discovered VFIO device should stay as VFIO")
	assert.Nil(t, allocDev.AmdGpu, "should not gain an AmdGpu entry")
}

func TestRestoreFromVfio_NilMap(t *testing.T) {
	state := &DeviceState{
		allocatable: AllocatableDevices{
			"gpu-vfio-0": {Vfio: &AmdGpuVFIOInfo{PCIAddress: "0000:0d:00.0"}},
		},
	}

	assert.NotPanics(t, func() {
		state.restoreFromVfio("gpu-vfio-0")
	})
}

func TestUnprepareDevices_RestoresConvertedDevice(t *testing.T) {
	original := &AmdGpuInfo{PCIAddress: "0000:0d:00.0", cardIndex: 0, renderIndex: 128}
	state := &DeviceState{
		allocatable: AllocatableDevices{
			"gpu-0-128": {Vfio: &AmdGpuVFIOInfo{
				PCIAddress:         "0000:0d:00.0",
				preConfigureDriver: "vfio-pci",
			}},
		},
		claimVfioConversions: map[string]*AmdGpuInfo{
			"gpu-0-128": original,
		},
		vfioManager: &VfioPciManager{},
	}
	devices := PreparedDevices{
		{Device: drapbv1.Device{DeviceName: "gpu-0-128"}},
	}

	err := state.unprepareDevices("test-claim", devices)
	assert.NoError(t, err)

	allocDev := state.allocatable["gpu-0-128"]
	assert.NotNil(t, allocDev.AmdGpu, "AmdGpu should be restored after unprepare")
	assert.Nil(t, allocDev.Vfio, "Vfio should be cleared after unprepare")
	assert.Equal(t, consts.AmdGpuDeviceType, allocDev.Type())
}

func TestUnprepareDevices_PreDiscoveredVfioNotRestored(t *testing.T) {
	state := &DeviceState{
		allocatable: AllocatableDevices{
			"gpu-vfio-0": {Vfio: &AmdGpuVFIOInfo{
				PCIAddress:         "0000:0d:00.0",
				preConfigureDriver: "vfio-pci",
			}},
		},
		claimVfioConversions: map[string]*AmdGpuInfo{},
		vfioManager:          &VfioPciManager{},
	}
	devices := PreparedDevices{
		{Device: drapbv1.Device{DeviceName: "gpu-vfio-0"}},
	}

	err := state.unprepareDevices("test-claim", devices)
	assert.NoError(t, err)

	allocDev := state.allocatable["gpu-vfio-0"]
	assert.NotNil(t, allocDev.Vfio, "pre-discovered VFIO should stay as VFIO")
	assert.Nil(t, allocDev.AmdGpu, "should not gain an AmdGpu entry")
}

func TestPreparedDevicesGetDevices(t *testing.T) {
	tests := map[string]struct {
		preparedDevices PreparedDevices
		expected        []*drapbv1.Device
	}{
		"nil PreparedDevices": {
			preparedDevices: nil,
			expected:        nil,
		},
		"several PreparedDevices": {
			preparedDevices: PreparedDevices{
				{Device: drapbv1.Device{DeviceName: "dev1"}},
				{Device: drapbv1.Device{DeviceName: "dev2"}},
				{Device: drapbv1.Device{DeviceName: "dev3"}},
			},
			expected: []*drapbv1.Device{
				{DeviceName: "dev1"},
				{DeviceName: "dev2"},
				{DeviceName: "dev3"},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			devices := test.preparedDevices.GetDevices()
			assert.Equal(t, test.expected, devices)
		})
	}
}

// The per-claim CDI spec files live in --cdi-root (/var/run/cdi in production),
// a tmpfs, while the checkpoint is on disk and survives a reboot. After a reboot
// kubelet restores its state with the CDI device IDs it already computed and
// never calls Prepare again, so the plugin must rewrite those specs out of the
// checkpoint at startup — otherwise every container of the claim fails with
// "unresolvable CDI devices" and recreating the pods or the DaemonSet does not
// help.
func TestRestoreCheckpointStateRewritesClaimSpecs(t *testing.T) {
	const claimUID = "claim-uid-1"

	cdiRoot := t.TempDir()
	pluginsDir := t.TempDir()

	handler, err := NewCDIHandler(&Config{flags: &Flags{cdiRoot: cdiRoot, nodeName: "node-1"}})
	require.NoError(t, err)

	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", "0:0-51", 8<<30)
	slot := 1
	device := &PreparedDevice{
		Device: drapbv1.Device{
			RequestNames: []string{"req-0"},
			PoolName:     "node-1",
			DeviceName:   "gpu-0-128",
		},
		ContainerEdits: &cdiapi.ContainerEdits{ContainerEdits: &cdispec.ContainerEdits{
			DeviceNodes: []*cdispec.DeviceNode{{
				Path:        "/dev/dri/renderD128",
				HostPath:    "/dev/dri/renderD128",
				Type:        "c",
				Major:       226,
				Minor:       128,
				Permissions: "rwm",
			}},
			Env:   env,
			Hooks: hooks,
		}},
		// A partition slot makes the CDI device name carry the slot, which is the
		// case a synthetic-partition claim needs restored after a reboot.
		PartitionSlot: &slot,
	}
	// What Prepare recorded: after a restart the kubelet asks for exactly these IDs.
	device.CdiDeviceIds = handler.GetClaimDevices(claimUID, []*PreparedDevice{device})

	checkpoint := newCheckpoint()
	checkpoint.V1.PreparedClaims[claimUID] = PreparedDevices{device}
	// A claim recorded without devices must not stop the plugin from starting:
	// CDI rejects an empty spec, and Prepare re-drives such an entry itself.
	checkpoint.V1.PreparedClaims["claim-without-devices"] = nil
	manager, err := checkpointmanager.NewCheckpointManager(pluginsDir)
	require.NoError(t, err)
	require.NoError(t, manager.CreateCheckpoint(DriverPluginCheckpointFile, checkpoint))

	// A fresh boot: the checkpoint survived, the tmpfs-backed CDI root is empty.
	specPath := filepath.Join(cdiRoot, cdiapi.GenerateTransientSpecName(cdiVendor, cdiClass, claimUID)+".yaml")
	_, err = os.Stat(specPath)
	require.True(t, os.IsNotExist(err), "the claim spec must be absent before the plugin starts, as after a reboot")

	state := &DeviceState{cdi: handler, checkpointManager: manager}
	require.NoError(t, state.restoreCheckpointState())

	regenerated, err := cdiapi.ReadSpec(specPath, 0)
	require.NoError(t, err, "the claim spec file must be written back at startup")
	require.Len(t, regenerated.Devices, 1)
	assert.Equal(t, claimUID+"-gpu-0-128-slot1", regenerated.Devices[0].Name)
	assert.Equal(t, []string{"k8s.gpu.amd.com/gpu=common", "k8s.gpu.amd.com/gpu=" + claimUID + "-gpu-0-128-slot1"}, device.CdiDeviceIds,
		"the regenerated name must be the one the checkpoint records, or kubelet asks for a device the spec does not declare")

	edits := regenerated.Devices[0].ContainerEdits
	assert.Contains(t, edits.Env, "HSA_CU_MASK=0:0-51")
	require.Len(t, edits.DeviceNodes, 1)
	assert.Equal(t, int64(226), edits.DeviceNodes[0].Major)
	assert.Equal(t, int64(128), edits.DeviceNodes[0].Minor)
	require.Len(t, edits.Hooks, 1)
	assert.Equal(t, "/run/cdi/dmem-hook", edits.Hooks[0].Path)
	assert.Contains(t, edits.Hooks[0].Env, "AMD_GPU_PCI_BDF=0000:e3:00.0")
	assert.Contains(t, edits.Hooks[0].Env, "AMD_GPU_MEM_BYTES=8589934592")

	// A plain plugin restart rewrites the same bytes, so the CDI root is unchanged.
	first, err := os.ReadFile(specPath)
	require.NoError(t, err)
	require.NoError(t, state.restoreCheckpointState())
	second, err := os.ReadFile(specPath)
	require.NoError(t, err)
	assert.Equal(t, string(first), string(second), "restoring an already-correct CDI root must be a no-op")

	// A further reboot wipes the root again; the next start must still recover it.
	require.NoError(t, os.Remove(specPath))
	require.NoError(t, state.restoreCheckpointState())
	third, err := os.ReadFile(specPath)
	require.NoError(t, err)
	assert.Equal(t, string(first), string(third))
}

// The restore above is only a fix if startup actually calls it. This drives the
// real NewDeviceState entry point with a checkpoint already on disk, which is the
// only place the wiring can be observed (a missing call there leaves every other
// test green).
func TestNewDeviceStateRestoresClaimSpecsFromCheckpoint(t *testing.T) {
	const (
		nodeName = "node-1"
		claimUID = "claim-uid-1"
	)

	cdiRoot := t.TempDir()
	pluginsDir := t.TempDir()

	checkpoint := newCheckpoint()
	checkpoint.V1.PreparedClaims[claimUID] = PreparedDevices{{
		Device: drapbv1.Device{
			RequestNames: []string{"req-0"},
			PoolName:     nodeName,
			DeviceName:   "gpu-0-128",
		},
		ContainerEdits: &cdiapi.ContainerEdits{ContainerEdits: &cdispec.ContainerEdits{
			Env: []string{"HSA_CU_MASK=0:0-51"},
		}},
	}}
	manager, err := checkpointmanager.NewCheckpointManager(filepath.Join(pluginsDir, consts.DriverName))
	require.NoError(t, err)
	require.NoError(t, manager.CreateCheckpoint(DriverPluginCheckpointFile, checkpoint))

	state, err := NewDeviceState(&Config{flags: &Flags{
		cdiRoot:                     cdiRoot,
		nodeName:                    nodeName,
		kubeletPluginsDirectoryPath: pluginsDir,
	}})
	require.NoError(t, err)
	require.NotNil(t, state)

	specPath := filepath.Join(cdiRoot, cdiapi.GenerateTransientSpecName(cdiVendor, cdiClass, claimUID)+".yaml")
	_, err = os.Stat(specPath)
	assert.NoError(t, err, "NewDeviceState must regenerate the checkpointed claims' CDI specs")
}

// TestPartitionSharesForClaim_DriverAndPoolMismatch guards the identity check
// added to partitionSharesForClaim: a DRA device is identified by
// (driver, pool, device), not device name alone. A same-named result from
// another driver or another pool must not be treated as a local partition
// device, even though the name collides with a real local synthetic device.
func TestPartitionSharesForClaim_DriverAndPoolMismatch(t *testing.T) {
	const (
		nodeName   = "node-a"
		deviceName = "gpu-0-cpx-nps4"
	)
	allocatable := AllocatableDevices{
		deviceName: {SyntheticPartition: &SyntheticPartitionDevice{GPUIndex: 0}},
	}

	newClaim := func(driver, pool string) *resourceapi.ResourceClaim {
		return &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{UID: types.UID("claim-1")},
			Status: resourceapi.ResourceClaimStatus{
				Allocation: &resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Request: "gpu", Driver: driver, Pool: pool, Device: deviceName},
						},
					},
				},
			},
		}
	}

	tests := map[string]struct {
		driver      string
		pool        string
		expectEmpty bool
	}{
		"matching driver and pool": {
			driver: consts.DriverName, pool: nodeName, expectEmpty: false,
		},
		"other driver, same pool": {
			driver: "other-driver.example.com", pool: nodeName, expectEmpty: true,
		},
		"local driver, other pool": {
			driver: consts.DriverName, pool: "node-b", expectEmpty: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			shares := partitionSharesForClaim(newClaim(test.driver, test.pool), allocatable, nodeName)
			if test.expectEmpty {
				assert.Empty(t, shares, "result from a mismatched driver/pool must not be treated as a local partition device")
			} else {
				assert.Len(t, shares, 1)
			}
		})
	}
}
