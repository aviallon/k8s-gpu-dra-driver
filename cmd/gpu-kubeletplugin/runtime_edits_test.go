/*
Copyright 2026 The Kubernetes Authors.

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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ROCm/k8s-gpu-dra-driver/pkg/consts"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"

	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdispec "tags.cncf.io/container-device-interface/specs-go"
)

// testDevice builds a whole MI210-class GPU (104 CUs, 64 GiB).
func testDevice() *AllocatableDevice {
	return &AllocatableDevice{AmdGpu: &AmdGpuInfo{
		PCIAddress:   "0000:e3:00.0",
		MemoryBytes:  68702699520,
		ComputeUnits: 104,
	}}
}

// resultWith builds an allocation result carrying the given ConsumedCapacity.
func resultWith(cap map[resourceapi.QualifiedName]string) *resourceapi.DeviceRequestAllocationResult {
	cc := map[resourceapi.QualifiedName]resource.Quantity{}
	for k, v := range cap {
		cc[k] = resource.MustParse(v)
	}
	return &resourceapi.DeviceRequestAllocationResult{ConsumedCapacity: cc}
}

func TestCuMaskFromCapacity(t *testing.T) {
	dev := testDevice()
	alloc := newCUAllocator()

	// No computeUnits capacity = 100% of the GPU -> no mask.
	assert.Equal(t, "", cuMaskFromCapacity(resultWith(nil), dev, alloc, 0, "s0"))
	// Whole-device worth = 100% -> no mask.
	assert.Equal(t, "", cuMaskFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"computeUnits": "104"}), dev, alloc, 0, "s1"))
	// A real fraction -> a disjoint N-CU range.
	mask := cuMaskFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"computeUnits": "13"}), dev, alloc, 0, "s2")
	assert.Equal(t, "0:0-12", mask)
}

func TestCUAllocatorDisjoint(t *testing.T) {
	dev := testDevice()
	alloc := newCUAllocator()
	a := cuMaskFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"computeUnits": "52"}), dev, alloc, 0, "share-a")
	b := cuMaskFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"computeUnits": "52"}), dev, alloc, 0, "share-b")
	assert.Equal(t, "0:0-51", a)
	assert.Equal(t, "0:52-103", b)
	// Idempotent: re-assigning the same share returns the same range.
	assert.Equal(t, a, cuMaskFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"computeUnits": "52"}), dev, alloc, 0, "share-a"))
	// After release, the range is reusable.
	alloc.release("share-a")
	assert.Equal(t, "0:0-51", cuMaskFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"computeUnits": "52"}), dev, alloc, 0, "share-c"))
}

func TestDmemBudgetFromCapacity(t *testing.T) {
	dev := testDevice()
	// No memory capacity = 100% -> no cap (0).
	assert.Equal(t, uint64(0), dmemBudgetFromCapacity(resultWith(nil), dev))
	// Whole-device worth = 100% -> no cap.
	assert.Equal(t, uint64(0), dmemBudgetFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"memory": "64Gi"}), dev))
	// A real fraction -> the bytes.
	assert.Equal(t, uint64(8589934592), dmemBudgetFromCapacity(resultWith(map[resourceapi.QualifiedName]string{"memory": "8Gi"}), dev))
}

// TestWholeGpuNoMaskNoCap: a whole-GPU claim (no ConsumedCapacity, e.g. vLLM via
// amd.com/gpu) must get NO HSA_CU_MASK and NO dmem hook — the full device.
func TestWholeGpuNoMaskNoCap(t *testing.T) {
	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", "", 0)
	assert.Empty(t, env)
	assert.Empty(t, hooks)
}

// TestBuildContainerRuntimeEdits: a fractional claim gets both the CU-mask env
// and the createRuntime dmem hook with the exact payloads.
func TestBuildContainerRuntimeEdits(t *testing.T) {
	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", "0:52-103", 8589934592)

	require.Len(t, hooks, 1)
	assert.Equal(t, "createRuntime", hooks[0].HookName)
	assert.Equal(t, "/run/cdi/dmem-hook", hooks[0].Path)
	assert.Equal(t, []string{
		"AMD_GPU_PCI_BDF=0000:e3:00.0",
		"AMD_GPU_MEM_BYTES=8589934592",
	}, hooks[0].Env)
	assert.Equal(t, []string{"HSA_CU_MASK=0:52-103"}, env)

	rendered, err := json.MarshalIndent(cdispec.ContainerEdits{Env: env, Hooks: hooks}, "", "  ")
	require.NoError(t, err)
	t.Logf("example CDI ContainerEdits:\n%s", rendered)
	assert.Contains(t, string(rendered), `"hookName": "createRuntime"`)
	assert.Contains(t, string(rendered), `"HSA_CU_MASK=0:52-103"`)
}

// TestBuildContainerRuntimeEditsSkipsHookWithoutBudget: a CU fraction with no
// memory budget -> the mask stays, the hook is omitted (a 0-byte cap would
// break the workload).
func TestBuildContainerRuntimeEditsSkipsHookWithoutBudget(t *testing.T) {
	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", "0:0-12", 0)
	assert.Equal(t, []string{"HSA_CU_MASK=0:0-12"}, env)
	assert.Empty(t, hooks)
}

// TestCreateClaimSpecFileCarriesRuntimeEdits drives the REAL spec-file path
// (CreateClaimSpecFile -> CDI cache -> disk -> ReadSpec) and asserts the hook and
// CU mask survive it — the file on disk is what the runtime consumes.
func TestCreateClaimSpecFileCarriesRuntimeEdits(t *testing.T) {
	cdiRoot := t.TempDir()
	handler, err := NewCDIHandler(&Config{flags: &Flags{cdiRoot: cdiRoot, nodeName: "node-1"}})
	require.NoError(t, err)

	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", "0:0-51", 8589934592)
	devices := PreparedDevices{
		{
			Device: drapbv1.Device{DeviceName: "gpu-1-2"},
			ContainerEdits: &cdiapi.ContainerEdits{ContainerEdits: &cdispec.ContainerEdits{
				Env:   env,
				Hooks: hooks,
			}},
		},
	}
	require.NoError(t, handler.CreateClaimSpecFile("claim-uid-1", devices))

	entries, err := os.ReadDir(cdiRoot)
	require.NoError(t, err)
	require.Len(t, entries, 1)
	spec, err := cdiapi.ReadSpec(filepath.Join(cdiRoot, entries[0].Name()), 0)
	require.NoError(t, err)
	require.Len(t, spec.Devices, 1)

	edits := spec.Devices[0].ContainerEdits
	assert.Contains(t, edits.Env, "HSA_CU_MASK=0:0-51")
	require.Len(t, edits.Hooks, 1)
	assert.Equal(t, "createRuntime", edits.Hooks[0].HookName)
	assert.Equal(t, "/run/cdi/dmem-hook", edits.Hooks[0].Path)
}

func TestBuildGPUIndexByPCI(t *testing.T) {
	alloc := AllocatableDevices{
		"gpu-0": &AllocatableDevice{AmdGpu: &AmdGpuInfo{PCIAddress: "0000:83:00.0"}},
		"gpu-1": &AllocatableDevice{AmdGpu: &AmdGpuInfo{PCIAddress: "0000:03:00.0"}},
		"gpu-2": &AllocatableDevice{AmdGpu: &AmdGpuInfo{PCIAddress: "0000:43:00.0"}},
	}
	byPCI := buildGPUIndexByPCI(alloc)
	assert.Equal(t, 0, byPCI["0000:03:00.0"]) // sorted BDF order
	assert.Equal(t, 1, byPCI["0000:43:00.0"])
	assert.Equal(t, 2, byPCI["0000:83:00.0"])
	_ = consts.DriverName // keep the consts import if unused elsewhere
}
