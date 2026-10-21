/*
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	klog "k8s.io/klog/v2"

	cdispec "tags.cncf.io/container-device-interface/specs-go"
)

// Runtime quota edits: HSA_CU_MASK + the createRuntime dmem hook.
//
// WHY these exist on every allocated GPU device's CDI edits:
//
//   - HSA_CU_MASK confines the workload to a disjoint half of the GPU's
//     compute units so two workloads sharing the GPU cannot starve each
//     other's CUs.
//   - The createRuntime hook writes the kernel's per-region "dmem" VRAM cap
//     onto the POD's cgroup at container creation time. Nothing else runs at
//     that moment, so without the hook the pod has no VRAM quota and a
//     workload can exhaust the whole device's VRAM.
const (
	// dmemHookInstallPath is where the dmem-hook binary must be installed on
	// the HOST (OCI hooks run on the host, outside the workload's mount
	// namespace). It lives under /run (tmpfs) because it is a runtime artifact
	// the driver DaemonSet installs at startup — ephemeral, re-created each
	// boot — not persistent add-on software (which is what /opt is for).
	// Deployment places cmd/dmem-hook's output there via a hostPath volume
	// (e.g. an init container copying /usr/bin/dmem-hook to /run/amd-gpu/).
	dmemHookInstallPath = "/run/amd-gpu/dmem-hook"

	// hookEnvPCIBDF / hookEnvMemBytes are the env contract between this
	// driver and cmd/dmem-hook; they must match cmd/dmem-hook's envPCIBDF /
	// envMemBytes constants.
	hookEnvPCIBDF   = "AMD_GPU_PCI_BDF"
	hookEnvMemBytes = "AMD_GPU_MEM_BYTES"

	// hsaCuMaskEnv is the ROCm runtime env var selecting compute units.
	hsaCuMaskEnv = "HSA_CU_MASK"

	// totalCUs is the CU count the 52-CU split is defined over (104 CUs, e.g.
	// MI210-class ASICs; MI300X-class XCDs have a different count). The
	// prescribed rule is a 52-CU half of 104; ASICs with other CU counts need
	// per-device derivation (from AmdGpuInfo.ComputeUnits) — see the report.
	totalCUs = 104
	// cuHalf splits the CUs into two disjoint halves of 52.
	cuHalf = totalCUs / 2

	// memoryResourceName is the ConsumedCapacity key the scheduler records for
	// VRAM (matches the "memory" device capacity in deviceinfo.go).
	memoryResourceName resourceapi.QualifiedName = "memory"
)

// cuMaskForGPUIndex returns the HSA_CU_MASK value that confines a container to
// one disjoint 52-CU half of the GPU's 104 CUs.
//
// Mapping rule (deterministic per GPU):
//
//	gpuIndex even -> "0:0-51"    (lower half: CUs 0-51)
//	gpuIndex odd  -> "0:52-103"  (upper half: CUs 52-103)
//
// where gpuIndex is the GPU's ordinal among the node's GPU PCI addresses in
// sorted BDF order (see buildGPUIndexByPCI). The halves are disjoint BETWEEN
// GPUs; two shares of the SAME GPU resolve to the same GPU index and therefore
// receive the same half (the rule is per-GPU by design — see the report's
// known-limitations section). The leading "0:" prefix follows the prescribed
// <idx>:<lo>-<hi> format (verify against the pinned ROCm runtime before merge
// — see report).
func cuMaskForGPUIndex(gpuIndex int) string {
	if gpuIndex%2 == 0 {
		return fmt.Sprintf("0:0-%d", cuHalf-1)
	}
	return fmt.Sprintf("0:%d-%d", cuHalf, totalCUs-1)
}

// buildGPUIndexByPCI maps every allocatable device's PCI BDF to its GPU index:
// the ordinal of the BDF in the sorted list of distinct GPU BDFs on the node.
// enumerateAllPossibleDevices assigns SyntheticPartitionDevice.GPUIndex by the
// same sorted-PCI order, so both agree on which GPU is "even" and which is
// "odd". Built once at startup so the CU-mask split is stable for the
// lifetime of the plugin.
func buildGPUIndexByPCI(allocatable AllocatableDevices) map[string]int {
	seen := make(map[string]struct{})
	for _, dev := range allocatable {
		if bdf := dev.GetPCIAddress(); bdf != "" {
			seen[bdf] = struct{}{}
		}
	}
	byPCI := make(map[string]int, len(seen))
	for i, bdf := range slices.Sorted(maps.Keys(seen)) {
		byPCI[bdf] = i
	}
	return byPCI
}

// deviceMemoryBudgetBytes returns this claim's VRAM budget for the device in
// bytes, or 0 when no meaningful cap can be derived ("no cap").
//
// Preference order:
//  1. the allocation result's ConsumedCapacity["memory"] when the key is
//     present — the scheduler's number is authoritative, EVEN when it is zero
//     (shared devices report zero-consumption entries); an explicit zero must
//     NOT silently inflate to the whole device's VRAM;
//  2. otherwise the device's full MemoryBytes.
//
// A return of 0 means "no cap": buildContainerRuntimeEdits emits no dmem hook
// for it (a 0-byte cap would break every workload), and applyConfig logs it.
func deviceMemoryBudgetBytes(result *resourceapi.DeviceRequestAllocationResult, dev *AllocatableDevice) uint64 {
	if result != nil {
		if qty, ok := result.ConsumedCapacity[memoryResourceName]; ok {
			if v := qty.Value(); v > 0 {
				return uint64(v)
			}
			// Present but zero/negative: explicitly no budget.
			return 0
		}
	}
	if dev != nil {
		return dev.MemoryBytes()
	}
	return 0
}

// buildContainerRuntimeEdits returns the per-device Env and Hooks every
// allocated GPU device's CDI edits carry:
//
//   - Env: HSA_CU_MASK=<idx>:<lo>-<hi> selecting a disjoint CU half.
//   - Hooks: a createRuntime hook that runs dmem-hook on the host with the
//     GPU's BDF and the claim's VRAM budget; the hook writes
//     "drm/<bdf>/vram <bytes>" to dmem.max on the pod's cgroup so the kernel
//     enforces the cap for the pod's whole lifetime.
//
// When memBudgetBytes is 0 ("no cap": consumedCapacity explicitly zero or
// VRAM unreadable at discovery) no hook is emitted — a 0-byte kernel cap
// would break every workload, and a hard container-start failure on degraded
// metadata would be worse than no cap. The HSA_CU_MASK env is independent and
// always emitted. Callers log when this happens (never silent).
func buildContainerRuntimeEdits(pciBDF string, memBudgetBytes uint64, gpuIndex int) (env []string, hooks []*cdispec.Hook) {
	env = []string{
		fmt.Sprintf("%s=%s", hsaCuMaskEnv, cuMaskForGPUIndex(gpuIndex)),
	}
	if memBudgetBytes == 0 {
		return env, nil
	}
	hooks = []*cdispec.Hook{
		{
			// "createRuntime" is the CDI hook name the CDI library translates
			// into an OCI createRuntime hook: it runs after the container is
			// created (pod cgroup exists) and before it starts.
			HookName: "createRuntime",
			Path:     dmemHookInstallPath,
			Env: []string{
				fmt.Sprintf("%s=%s", hookEnvPCIBDF, pciBDF),
				fmt.Sprintf("%s=%d", hookEnvMemBytes, memBudgetBytes),
			},
		},
	}
	return env, hooks
}

// checkDmemPrereqs logs loud, early warnings when the node is not ready for
// the dmem cap path: without the kernel dmem controller or without the hook
// binary installed on the host, every GPU container start would fail at its
// createRuntime hook. Surfacing it once here at plugin startup beats an
// outage discovered per-pod. Non-fatal by design: the plugin must still serve
// non-dmem nodes (where no cap is emitted and the hook is the only failure).
func checkDmemPrereqs() {
	const (
		cgroupControllers = "/sys/fs/cgroup/cgroup.controllers"
	)
	data, err := os.ReadFile(cgroupControllers)
	if err != nil {
		klog.Errorf("dmem VRAM caps unavailable: cannot read %s: %v", cgroupControllers, err)
	} else if !slices.Contains(strings.Fields(string(data)), "dmem") {
		klog.Errorf("dmem VRAM caps unavailable: kernel cgroup controller %q not in %s (got %q); GPU containers will fail at their createRuntime hook",
			"dmem", cgroupControllers, strings.TrimSpace(string(data)))
	}
	fi, err := os.Stat(dmemHookInstallPath)
	switch {
	case err != nil:
		klog.Errorf("dmem VRAM caps unavailable: hook binary %s missing on this host (install cmd/dmem-hook there); GPU containers will fail at their createRuntime hook: %v",
			dmemHookInstallPath, err)
	case fi.Mode()&0o111 == 0:
		klog.Errorf("dmem VRAM caps unavailable: hook binary %s is not executable (mode %s)", dmemHookInstallPath, fi.Mode())
	}
}
