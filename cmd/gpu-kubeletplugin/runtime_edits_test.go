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

// TestCuMaskForGPUIndex checks the documented mapping: even GPU index ->
// lower CU half, odd GPU index -> upper CU half.
func TestCuMaskForGPUIndex(t *testing.T) {
	assert.Equal(t, "0:0-51", cuMaskForGPUIndex(0))
	assert.Equal(t, "0:52-103", cuMaskForGPUIndex(1))
	assert.Equal(t, "0:0-51", cuMaskForGPUIndex(2))
	assert.Equal(t, "0:52-103", cuMaskForGPUIndex(3))
}

// TestBuildGPUIndexByPCI checks the deterministic GPU index: ordinal of the
// PCI BDF in sorted order of the node's GPU BDFs.
func TestBuildGPUIndexByPCI(t *testing.T) {
	allocatable := AllocatableDevices{
		"gpu-1-2": {AmdGpu: &AmdGpuInfo{PCIAddress: "0000:e3:00.0"}},
		"gpu-3-4": {AmdGpu: &AmdGpuInfo{PCIAddress: "0000:03:00.0"}},
		// GPUIndex consistent with the map (same sorted-PCI order that
		// enumerateAllPossibleDevices assigns): "0000:81:00.0" -> 1.
		"gpu-5-6": {SyntheticPartition: &SyntheticPartitionDevice{PCIAddress: "0000:81:00.0", GPUIndex: 1}},
	}
	byPCI := buildGPUIndexByPCI(allocatable)
	assert.Equal(t, map[string]int{
		"0000:03:00.0": 0,
		"0000:81:00.0": 1,
		"0000:e3:00.0": 2,
	}, byPCI)
	// The map must agree with SyntheticPartitionDevice.GPUIndex (one source of
	// truth for "which GPU is even/odd").
	assert.Equal(t, allocatable["gpu-5-6"].SyntheticPartition.GPUIndex, byPCI["0000:81:00.0"])
}

// TestDeviceMemoryBudgetBytes checks the budget preference: the allocation
// result's ConsumedCapacity["memory"] wins when present; the device's full
// MemoryBytes is the fallback when absent.
func TestDeviceMemoryBudgetBytes(t *testing.T) {
	dev := &AllocatableDevice{AmdGpu: &AmdGpuInfo{MemoryBytes: 32 << 30}}

	// ConsumedCapacity present -> per-claim budget.
	result := &resourceapi.DeviceRequestAllocationResult{
		ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{
			memoryResourceName: resource.MustParse("8Gi"),
		},
	}
	assert.Equal(t, uint64(8<<30), deviceMemoryBudgetBytes(result, dev))

	// ConsumedCapacity absent -> fallback to full device memory.
	assert.Equal(t, uint64(32<<30), deviceMemoryBudgetBytes(&resourceapi.DeviceRequestAllocationResult{}, dev))

	// ConsumedCapacity present but zero (shared devices report zero-consumption
	// entries): the scheduler's number is authoritative -> no cap (0), NOT the
	// whole device's VRAM.
	zero := &resourceapi.DeviceRequestAllocationResult{
		ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{
			memoryResourceName: {},
		},
	}
	assert.Equal(t, uint64(0), deviceMemoryBudgetBytes(zero, dev))
}

// TestBuildContainerRuntimeEditsSkipsHookWithoutBudget: budget 0 = no cap ->
// the dmem hook must NOT be emitted (a 0-byte kernel cap would break every
// workload and a hard container-start failure on degraded metadata would be
// worse than no cap); HSA_CU_MASK is independent and stays.
func TestBuildContainerRuntimeEditsSkipsHookWithoutBudget(t *testing.T) {
	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", 0, 0)
	assert.Empty(t, hooks)
	assert.Equal(t, []string{"HSA_CU_MASK=0:0-51"}, env)
}

// TestBuildContainerRuntimeEdits renders the CDI edits and checks both the
// HSA_CU_MASK env and the createRuntime dmem hook are present with the exact
// expected payloads. Run with -v to see the rendered JSON.
func TestBuildContainerRuntimeEdits(t *testing.T) {
	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", 8589934592, 1)

	require.Len(t, hooks, 1)
	assert.Equal(t, "createRuntime", hooks[0].HookName)
	assert.Equal(t, "/opt/amd-gpu/dmem-hook", hooks[0].Path)
	assert.Equal(t, []string{
		"AMD_GPU_PCI_BDF=0000:e3:00.0",
		"AMD_GPU_MEM_BYTES=8589934592",
	}, hooks[0].Env)
	assert.Equal(t, []string{"HSA_CU_MASK=0:52-103"}, env)

	// Render exactly what ends up in the CDI spec (device ContainerEdits).
	rendered, err := json.MarshalIndent(cdispec.ContainerEdits{Env: env, Hooks: hooks}, "", "  ")
	require.NoError(t, err)
	t.Logf("example CDI ContainerEdits:\n%s", rendered)

	assert.Contains(t, string(rendered), `"hookName": "createRuntime"`)
	assert.Contains(t, string(rendered), `"HSA_CU_MASK=0:52-103"`)
	assert.Contains(t, string(rendered), `"AMD_GPU_PCI_BDF=0000:e3:00.0"`)
	assert.Contains(t, string(rendered), `"AMD_GPU_MEM_BYTES=8589934592"`)

	// Sanity: the device type constant path still builds for a synthetic
	// partition (the case GetPCIAddress previously missed).
	sp := &AllocatableDevice{SyntheticPartition: &SyntheticPartitionDevice{PCIAddress: "0000:81:00.0"}}
	assert.Equal(t, "0000:81:00.0", sp.GetPCIAddress())
	assert.Equal(t, consts.SyntheticPartitionDeviceType, sp.Type())
}

// TestCreateClaimSpecFileCarriesRuntimeEdits drives the REAL spec-file path
// (CreateClaimSpecFile -> CDI cache -> disk -> ReadSpec) and asserts the
// createRuntime hook and HSA_CU_MASK survive it — the file on disk is what the
// container runtime actually consumes.
func TestCreateClaimSpecFileCarriesRuntimeEdits(t *testing.T) {
	cdiRoot := t.TempDir()
	handler, err := NewCDIHandler(&Config{flags: &Flags{cdiRoot: cdiRoot, nodeName: "node-1"}})
	require.NoError(t, err)

	env, hooks := buildContainerRuntimeEdits("0000:e3:00.0", 8589934592, 0)
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
	assert.Equal(t, "/opt/amd-gpu/dmem-hook", edits.Hooks[0].Path)
	assert.Equal(t, []string{
		"AMD_GPU_PCI_BDF=0000:e3:00.0",
		"AMD_GPU_MEM_BYTES=8589934592",
	}, edits.Hooks[0].Env)
}
