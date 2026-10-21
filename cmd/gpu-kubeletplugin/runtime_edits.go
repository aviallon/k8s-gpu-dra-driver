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
	"bytes"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

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
	// dmemHookSourcePath is where the dmem-hook binary ships in the driver image;
	// dmemHookInstallPath is where it must be on the HOST for the CDI createRuntime
	// hook to exec it (OCI hooks run on the host, outside the workload's mount
	// namespace). The install path is under the driver's own /var/run/cdi hostPath
	// (the only writable /run mount the DaemonSet has), and /run is tmpfs (cleared
	// each boot), so installDmemHook copies the binary there at every plugin
	// start — self-contained and it survives reboots.
	dmemHookSourcePath  = "/usr/bin/dmem-hook"
	dmemHookInstallPath = "/run/cdi/dmem-hook"

	// hookEnvPCIBDF / hookEnvMemBytes are the env contract between this
	// driver and cmd/dmem-hook; they must match cmd/dmem-hook's envPCIBDF /
	// envMemBytes constants.
	hookEnvPCIBDF   = "AMD_GPU_PCI_BDF"
	hookEnvMemBytes = "AMD_GPU_MEM_BYTES"

	// hsaCuMaskEnv is the ROCm runtime env var selecting compute units.
	hsaCuMaskEnv = "HSA_CU_MASK"

	// memoryResourceName / computeUnitsResourceName are the ConsumedCapacity
	// keys the scheduler records for VRAM / compute units (matching the device
	// capacities in deviceinfo.go). They drive the dmem cap and the CU mask.
	memoryResourceName       resourceapi.QualifiedName = "memory"
	computeUnitsResourceName resourceapi.QualifiedName = "computeUnits"
)

// cuAssignment records one share's disjoint CU range on a GPU.
type cuAssignment struct {
	gpuIndex int
	lo, hi   int // inclusive CU range
}

// cuAllocator hands out disjoint compute-unit ranges to the shares of each GPU
// so two pods sharing one GPU never run on the same CUs. Ranges are contiguous
// (HSA_CU_MASK <idx>:<lo>-<hi>) and are released on Unprepare via the share key.
type cuAllocator struct {
	mu      sync.Mutex
	used    map[int][]bool // gpuIndex -> used[cu]
	byShare map[string]cuAssignment
}

func newCUAllocator() *cuAllocator {
	return &cuAllocator{used: map[int][]bool{}, byShare: map[string]cuAssignment{}}
}

// assign reserves n disjoint CUs on gpuIndex for shareKey and returns the
// inclusive range. Idempotent: re-assigning the same share returns its range.
// ok=false when the GPU has no free contiguous block of n CUs left.
func (a *cuAllocator) assign(gpuIndex, totalCUs, n int, shareKey string) (lo, hi int, ok bool) {
	if a == nil {
		return 0, 0, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if as, exists := a.byShare[shareKey]; exists {
		return as.lo, as.hi, true
	}
	used := a.used[gpuIndex]
	if len(used) < totalCUs {
		used = append(used, make([]bool, totalCUs-len(used))...)
		a.used[gpuIndex] = used
	}
	for lo = 0; lo+n <= totalCUs; lo++ {
		free := true
		for i := lo; i < lo+n; i++ {
			if used[i] {
				free = false
				break
			}
		}
		if free {
			for i := lo; i < lo+n; i++ {
				used[i] = true
			}
			a.byShare[shareKey] = cuAssignment{gpuIndex: gpuIndex, lo: lo, hi: lo + n - 1}
			return lo, lo + n - 1, true
		}
	}
	return 0, 0, false
}

// release frees shareKey's CU range so it can be reused by a later share.
func (a *cuAllocator) release(shareKey string) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	as, ok := a.byShare[shareKey]
	if !ok {
		return
	}
	if used := a.used[as.gpuIndex]; as.lo < len(used) {
		for i := as.lo; i <= as.hi && i < len(used); i++ {
			used[i] = false
		}
	}
	delete(a.byShare, shareKey)
}

// cuMaskFromCapacity derives the HSA_CU_MASK for a claim from its DRA
// ConsumedCapacity["computeUnits"] — the mask reflects what the scheduler
// actually allocated:
//   - no computeUnits capacity (or the whole device's worth) = 100% of the GPU
//     -> NO mask (the container sees every CU);
//   - a real fraction (0 < N < the device's CUs) -> a disjoint N-CU range from
//     alloc (so shares of one GPU never overlap).
//
// Mask format "<idx>:<lo>-<hi>" (idx = ROCm device ordinal; 0 on a container
// granted a single GPU).
func cuMaskFromCapacity(result *resourceapi.DeviceRequestAllocationResult, dev *AllocatableDevice, alloc *cuAllocator, gpuIndex int, shareKey string) string {
	if result == nil || dev == nil {
		return ""
	}
	totalCUs := int(dev.ComputeUnits())
	qty, ok := result.ConsumedCapacity[computeUnitsResourceName]
	if !ok {
		return "" // no capacity = 100% of the GPU
	}
	n := int(qty.Value())
	if n <= 0 || totalCUs <= 0 || n >= totalCUs {
		return "" // 100% (or unusable) = no mask
	}
	lo, hi, ok := alloc.assign(gpuIndex, totalCUs, n, shareKey)
	if !ok {
		klog.Errorf("no free %d-CU block on GPU %d for share %s; HSA_CU_MASK omitted", n, gpuIndex, shareKey)
		return ""
	}
	return fmt.Sprintf("0:%d-%d", lo, hi)
}

// dmemBudgetFromCapacity returns the claim's VRAM cap from
// ConsumedCapacity["memory"]: a real fraction (0 < M < the device's VRAM)
// returns the bytes; no capacity (or the whole device, or an explicit zero)
// returns 0 = "no cap" (no dmem hook) = 100% of the GPU.
func dmemBudgetFromCapacity(result *resourceapi.DeviceRequestAllocationResult, dev *AllocatableDevice) uint64 {
	if result == nil || dev == nil {
		return 0
	}
	qty, ok := result.ConsumedCapacity[memoryResourceName]
	if !ok {
		return 0 // no capacity = 100% of the GPU, no cap
	}
	v := qty.Value()
	if v <= 0 || uint64(v) >= dev.MemoryBytes() {
		return 0 // whole-device (or explicit zero) = no cap
	}
	return uint64(v)
}

// buildContainerRuntimeEdits returns the per-device Env and Hooks for a claim's
// CDI edits, both derived from its DRA ConsumedCapacity:
//   - Env HSA_CU_MASK: only when a CU fraction is requested (100% = no mask).
//   - Hooks createRuntime: writes the dmem VRAM cap, only when a memory
//     fraction is requested (100% = no hook).
//
// Whole-GPU claims (e.g. vLLM via amd.com/gpu, no capacity) get neither, so the
// full device is untouched.
func buildContainerRuntimeEdits(pciBDF, cuMask string, memBudgetBytes uint64) (env []string, hooks []*cdispec.Hook) {
	if cuMask != "" {
		env = []string{fmt.Sprintf("%s=%s", hsaCuMaskEnv, cuMask)}
	}
	if memBudgetBytes == 0 {
		return env, nil
	}
	hooks = []*cdispec.Hook{
		{
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

// installDmemHook copies the dmem-hook binary from the driver image
// (dmemHookSourcePath) onto the host at dmemHookInstallPath, so the CDI
// createRuntime hook can exec it. /run is tmpfs (cleared each boot), so this
// runs at every plugin start and makes the hook survive reboots without any
// DaemonSet init container. Idempotent: it overwrites only when the source
// differs, and the write is atomic (tmp + rename).
func installDmemHook() error {
	src, err := os.ReadFile(dmemHookSourcePath)
	if err != nil {
		return fmt.Errorf("read hook source %s: %w", dmemHookSourcePath, err)
	}
	if dst, err := os.ReadFile(dmemHookInstallPath); err == nil && bytes.Equal(dst, src) {
		return nil // already installed and current
	}
	if err := os.MkdirAll(filepath.Dir(dmemHookInstallPath), 0o755); err != nil {
		return fmt.Errorf("mkdir hook dir: %w", err)
	}
	tmp := dmemHookInstallPath + ".tmp"
	if err := os.WriteFile(tmp, src, 0o755); err != nil {
		return fmt.Errorf("write hook %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dmemHookInstallPath); err != nil {
		return fmt.Errorf("install hook %s: %w", dmemHookInstallPath, err)
	}
	klog.Infof("installed dmem-hook %s -> %s", dmemHookSourcePath, dmemHookInstallPath)
	return nil
}

// checkDmemPrereqs installs the dmem-hook binary then logs loud, early warnings
// when the node is not ready for the dmem cap path: without the kernel dmem
// controller or without the hook binary on the host, every GPU container start
// would fail at its createRuntime hook. Surfacing it once here at plugin startup
// beats an outage discovered per-pod. Non-fatal by design: the plugin must still
// serve non-dmem nodes (where no cap is emitted and the hook is the only failure).
func checkDmemPrereqs() {
	if err := installDmemHook(); err != nil {
		klog.Errorf("dmem-hook install failed (GPU containers needing a VRAM cap will fail at their createRuntime hook): %v", err)
	}
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
