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
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	testBDF        = "0000:e3:00.0"
	testMemBytes   = "8589934592"
	testWantDmem   = "drm/0000:e3:00.0/vram 8589934592\n"
	testPodSlice   = "containers-kubepods-burstable-pod123abc.slice"
	testContainerC = "cri-containerd-0123456789abcdef.scope"
)

// fakePodTree creates a temp cgroup tree shaped like the node's systemd
// hierarchy and returns the pod slice dir and the container scope dir inside it.
func fakePodTree(t *testing.T) (podSlice, containerScope string) {
	t.Helper()
	root := t.TempDir()
	podSlice = filepath.Join(root,
		"containers.slice",
		"containers-kubepods.slice",
		"containers-kubepods-burstable.slice",
		testPodSlice)
	containerScope = filepath.Join(podSlice, testContainerC)
	if err := os.MkdirAll(containerScope, 0o755); err != nil {
		t.Fatal(err)
	}
	return podSlice, containerScope
}

// fakeState writes a bundle dir with config.json carrying cgroupsPath and
// returns the marshalled OCI state JSON to feed run() on stdin.
func fakeState(t *testing.T, cgroupsPath string) (stateJSON []byte, bundle string) {
	t.Helper()
	bundle = t.TempDir()
	cfg := map[string]any{
		"linux": map[string]any{"cgroupsPath": cgroupsPath},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundle, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	state := ociState{ID: "test-container", PID: 4242, Bundle: bundle}
	stateJSON, err = json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	return stateJSON, bundle
}

func runHook(t *testing.T, stateJSON []byte) (string, string, error) {
	t.Helper()
	return runHookWith(t, stateJSON, "/sys/fs/cgroup", "/proc")
}

func runHookWith(t *testing.T, stateJSON []byte, cgroupRoot, procRoot string) (string, string, error) {
	t.Helper()
	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	err := run(bytes.NewReader(stateJSON), stdout, stderr, cgroupRoot, procRoot)
	return stdout.String(), stderr.String(), err
}

// TestRunWritesDmemMaxOnPodSlice is the end-to-end check: OCI state on stdin
// and a bundle config.json must produce exactly one dmem.max line on the pod
// slice — not on the container's own scope.
func TestRunWritesDmemMaxOnPodSlice(t *testing.T) {
	podSlice, containerScope := fakePodTree(t)
	stateJSON, _ := fakeState(t, containerScope)

	t.Setenv(envPCIBDF, testBDF)
	t.Setenv(envMemBytes, testMemBytes)

	stdout, stderr, err := runHook(t, stateJSON)
	if err != nil {
		t.Fatalf("run() failed: %v\nstderr: %s", err, stderr)
	}

	got, err := os.ReadFile(filepath.Join(podSlice, dmemFile))
	if err != nil {
		t.Fatalf("dmem.max not written on pod slice: %v", err)
	}
	if string(got) != testWantDmem {
		t.Errorf("dmem.max content = %q, want %q", got, testWantDmem)
	}
	if _, err := os.Stat(filepath.Join(containerScope, dmemFile)); !os.IsNotExist(err) {
		t.Errorf("dmem.max must NOT be written on the container scope (err=%v)", err)
	}
	if !strings.Contains(stdout, testBDF) {
		t.Errorf("stdout should attribute the write to %s, got %q", testBDF, stdout)
	}
}

// TestRunIdempotent: re-running the hook (second container of the same pod,
// or a retried start) rewrites the same line rather than failing or appending.
func TestRunIdempotent(t *testing.T) {
	podSlice, containerScope := fakePodTree(t)
	stateJSON, _ := fakeState(t, containerScope)
	t.Setenv(envPCIBDF, testBDF)
	t.Setenv(envMemBytes, testMemBytes)

	for i := 0; i < 3; i++ {
		if _, stderr, err := runHook(t, stateJSON); err != nil {
			t.Fatalf("run %d failed: %v\nstderr: %s", i, err, stderr)
		}
	}
	got, err := os.ReadFile(filepath.Join(podSlice, dmemFile))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != testWantDmem {
		t.Errorf("after 3 runs dmem.max = %q, want %q", got, testWantDmem)
	}
}

// TestRunFailsLoudWithoutPodSlice: a cgroup with no *-pod*.slice ancestor must
// exit with an error, never silently skip the cap.
func TestRunFailsLoudWithoutPodSlice(t *testing.T) {
	containerScope := filepath.Join(t.TempDir(), "not-a-pod.slice", testContainerC)
	if err := os.MkdirAll(containerScope, 0o755); err != nil {
		t.Fatal(err)
	}
	stateJSON, _ := fakeState(t, containerScope)
	t.Setenv(envPCIBDF, testBDF)
	t.Setenv(envMemBytes, testMemBytes)

	_, _, err := runHook(t, stateJSON)
	if err == nil {
		t.Fatal("run() must fail when no *-pod<uid>.slice ancestor exists")
	}
	if !strings.Contains(err.Error(), "no *-pod<uid>.slice ancestor") {
		t.Errorf("error should say no pod slice found, got: %v", err)
	}
	if !strings.Contains(err.Error(), "cgroupDriver=systemd") {
		t.Errorf("error should name the systemd-cgroup assumption, got: %v", err)
	}
}

// TestRunRejectsBadEnv: malformed BDF or byte count must fail, not write.
func TestRunRejectsBadEnv(t *testing.T) {
	podSlice, containerScope := fakePodTree(t)
	stateJSON, _ := fakeState(t, containerScope)

	for _, tc := range []struct{ bdf, mem string }{
		{"", testMemBytes},
		{"e3:00.0", testMemBytes}, // no domain
		{"not-a-bdf", testMemBytes},
		{testBDF, ""},
		{testBDF, "0"},
		{testBDF, "8Gi"},
	} {
		t.Setenv(envPCIBDF, tc.bdf)
		t.Setenv(envMemBytes, tc.mem)
		_, stderr, err := runHook(t, stateJSON)
		if err == nil {
			t.Errorf("bdf=%q mem=%q: run() must fail", tc.bdf, tc.mem)
		}
		_ = stderr
	}
	if _, err := os.Stat(filepath.Join(podSlice, dmemFile)); !os.IsNotExist(err) {
		t.Errorf("dmem.max must not be written on invalid input (err=%v)", err)
	}
}

// TestRunFallsBackToProcCgroup: when the bundle has no usable cgroupsPath, the
// hook resolves the cgroup from /proc/<pid>/cgroup. pid is taken from the OCI
// state; to keep the test hermetic it points at this test process's own /proc
// entry is not usable, so instead exercise the fallback through a state whose
// bundle is missing and verify the error mentions both attempts when /proc is
// unusable too.
func TestRunReportsBothResolutionAttempts(t *testing.T) {
	state := ociState{ID: "x", PID: 1 << 30, Bundle: filepath.Join(t.TempDir(), "missing-bundle")}
	data, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(envPCIBDF, testBDF)
	t.Setenv(envMemBytes, testMemBytes)

	_, _, err = runHook(t, data)
	if err == nil {
		t.Fatal("run() must fail loudly when the cgroup cannot be resolved")
	}
	// Fail-loud: the error must mention both resolution sources so the operator
	// can see what was tried.
	for _, want := range []string{"config.json", "/proc"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %q, got: %v", want, err)
		}
	}
}

// TestRunLowercasesBDF: sysfs-style region names are lowercase; a mixed-case
// env BDF must be normalized before it targets a region.
func TestRunLowercasesBDF(t *testing.T) {
	podSlice, containerScope := fakePodTree(t)
	stateJSON, _ := fakeState(t, containerScope)
	t.Setenv(envPCIBDF, "0000:E3:00.0")
	t.Setenv(envMemBytes, testMemBytes)

	if _, stderr, err := runHook(t, stateJSON); err != nil {
		t.Fatalf("run() failed: %v\nstderr: %s", err, stderr)
	}
	got, err := os.ReadFile(filepath.Join(podSlice, dmemFile))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != testWantDmem {
		t.Errorf("dmem.max content = %q, want %q (lowercased BDF)", got, testWantDmem)
	}
}

// TestRunProcFallbackSuccess: when the bundle has no usable cgroupsPath but
// procRoot/<pid>/cgroup resolves, the hook must use the /proc path (the
// fallback SUCCESS path, not just its failure diagnostics).
func TestRunProcFallbackSuccess(t *testing.T) {
	podSlice, containerScope := fakePodTree(t)

	// Bundle whose config.json has an empty cgroupsPath -> primary resolution fails.
	bundle := t.TempDir()
	if err := os.WriteFile(filepath.Join(bundle, "config.json"), []byte(`{"linux":{"cgroupsPath":""}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	state := ociState{ID: "x", PID: 4242, Bundle: bundle}
	stateJSON, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}

	// Fake procfs with the container's cgroup v2 line.
	procRoot := t.TempDir()
	procDir := filepath.Join(procRoot, "4242")
	if err := os.MkdirAll(procDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "cgroup"), []byte("0::"+containerScope+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv(envPCIBDF, testBDF)
	t.Setenv(envMemBytes, testMemBytes)
	stdout, stderr, err := runHookWith(t, stateJSON, "/sys/fs/cgroup", procRoot)
	if err != nil {
		t.Fatalf("run() failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stderr, "falling back to /proc") {
		t.Errorf("stderr should announce the /proc fallback, got %q", stderr)
	}
	if !strings.Contains(stdout, "(cgroup from /proc/<pid>/cgroup)") {
		t.Errorf("success line should attribute the /proc source, got %q", stdout)
	}
	got, err := os.ReadFile(filepath.Join(podSlice, dmemFile))
	if err != nil {
		t.Fatalf("dmem.max not written on pod slice via /proc fallback: %v", err)
	}
	if string(got) != testWantDmem {
		t.Errorf("dmem.max content = %q, want %q", got, testWantDmem)
	}
}

// TestCgroupDirFromPathDashedPodUID: a pod UID containing dashes expands to a
// chain of intermediate systemd slices (expandSlice semantics); the hook must
// resolve the final nested dir and still find the pod slice (the deepest one).
func TestCgroupDirFromPathDashedPodUID(t *testing.T) {
	const dashedSlice = "containers-kubepods-burstable-pod6d3f8f0a-1b2c-4d5e-8f90-abcdef012345.slice"
	root := t.TempDir()
	scopeDir := filepath.Join(root,
		"containers.slice",
		"containers-kubepods.slice",
		"containers-kubepods-burstable.slice",
		"containers-kubepods-burstable-pod6d3f8f0a.slice",
		"containers-kubepods-burstable-pod6d3f8f0a-1b2c.slice",
		"containers-kubepods-burstable-pod6d3f8f0a-1b2c-4d5e.slice",
		"containers-kubepods-burstable-pod6d3f8f0a-1b2c-4d5e-8f90.slice",
		dashedSlice,
		"cri-containerd-abc.scope")
	if err := os.MkdirAll(scopeDir, 0o755); err != nil {
		t.Fatal(err)
	}

	triple := dashedSlice + ":cri-containerd:abc"
	got, err := cgroupDirFromPath(triple, root)
	if err != nil {
		t.Fatalf("cgroupDirFromPath(%q): %v", triple, err)
	}
	if got != scopeDir {
		t.Errorf("cgroupDirFromPath(%q) = %q, want %q", triple, got, scopeDir)
	}

	// Walk-up must pick the DEEPEST matching slice (the full pod UID name).
	pod, err := findPodSlice(scopeDir)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(pod) != dashedSlice {
		t.Errorf("findPodSlice picked %q, want %q", filepath.Base(pod), dashedSlice)
	}
}

func TestExpandSlice(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"", ""},
		{"-", ""},
		{"containers-kubepods-burstable-pod123abc.slice",
			"/containers.slice/containers-kubepods.slice/containers-kubepods-burstable.slice/containers-kubepods-burstable-pod123abc.slice"},
		{"foo.slice", "/foo.slice"},
	} {
		got, err := expandSlice(tc.in)
		if err != nil {
			t.Errorf("expandSlice(%q) error: %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("expandSlice(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, bad := range []string{"nosuffix", "a-.slice", "-a.slice"} {
		if _, err := expandSlice(bad); err == nil {
			t.Errorf("expandSlice(%q) must fail", bad)
		}
	}
}

// TestCgroupDirFromPathTriple exercises the systemd "slice:prefix:name" form
// against a fake cgroup root.
func TestCgroupDirFromPathTriple(t *testing.T) {
	// Build the hierarchy under a root so cgroupDirFromPath's cgroupRoot join
	// lands inside it.
	root := t.TempDir()
	scopeDir := filepath.Join(root, "containers.slice", "containers-kubepods.slice",
		"containers-kubepods-burstable.slice", testPodSlice, "cri-containerd-abc.scope")
	if err := os.MkdirAll(scopeDir, 0o755); err != nil {
		t.Fatal(err)
	}

	triple := testPodSlice + ":cri-containerd:abc"
	got, err := cgroupDirFromPath(triple, root)
	if err != nil {
		t.Fatalf("cgroupDirFromPath(%q): %v", triple, err)
	}
	if got != scopeDir {
		t.Errorf("cgroupDirFromPath(%q) = %q, want %q", triple, got, scopeDir)
	}

	// An absolute cgroupfs path that exists is used as-is.
	absPod, _ := fakePodTree(t)
	got, err = cgroupDirFromPath(absPod, "/sys/fs/cgroup")
	if err != nil {
		t.Fatalf("cgroupDirFromPath(abs): %v", err)
	}
	if got != absPod {
		t.Errorf("absolute existing path should be used as-is, got %q", got)
	}
}

// TestFindPodSlice walks the fake hierarchy.
func TestFindPodSlice(t *testing.T) {
	podSlice, containerScope := fakePodTree(t)
	got, err := findPodSlice(containerScope)
	if err != nil {
		t.Fatal(err)
	}
	if got != podSlice {
		t.Errorf("findPodSlice(%q) = %q, want %q", containerScope, got, podSlice)
	}
}
