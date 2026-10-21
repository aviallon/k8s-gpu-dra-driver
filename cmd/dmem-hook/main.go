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

// Command dmem-hook is an OCI "createRuntime" hook installed on the host and
// attached by the AMD GPU DRA driver to every CDI device it allocates.
//
// WHY: when the kernel exposes per-region VRAM cgroup controllers ("dmem"),
// capping a pod's VRAM usage requires writing the budget to the POD's cgroup
// *at container creation time* — after kubelet has created the pod cgroup, but
// before the workload runs. A CDI createRuntime hook is the sanctioned CDI
// mechanism for that moment; it runs on the host inside the runtime's
// namespaces with the container's OCI state on stdin.
//
// It resolves the container's cgroup, walks up to the pod slice ancestor
// (the container's own cri-containerd-*.scope charges to the POD cgroup, so
// the cap must land on the pod slice), and writes one line to dmem.max:
//
//	drm/<AMD_GPU_PCI_BDF>/vram <AMD_GPU_MEM_BYTES>
//
// Environment (injected by the CDI hook entry the driver emits):
//
//	AMD_GPU_PCI_BDF    PCI BDF of the allocated GPU, e.g. "0000:e3:00.0"
//	AMD_GPU_MEM_BYTES  VRAM budget in bytes for this pod on that GPU
//
// Failure behaviour: any error (missing env, unresolvable cgroup, no pod
// slice ancestor, write failure) exits non-zero with a clear message on
// stderr so the runtime aborts the container start. It never silently skips
// the cap: a container running uncapped would be a silent quota bypass.
// Re-writes are idempotent (dmem.max entries are per-region; rewriting
// replaces the same region's line).
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// version is injected at build time via -ldflags "-X main.version=..." (see
// the repo Makefile's cmd-% rule).
var version = "dev"

const (
	// envPCIBDF names the env var carrying the GPU's PCI BDF.
	envPCIBDF = "AMD_GPU_PCI_BDF"
	// envMemBytes names the env var carrying the pod's VRAM budget in bytes.
	envMemBytes = "AMD_GPU_MEM_BYTES"

	// dmemFile is the per-region cgroup file the kernel exposes once the dmem
	// controller is enabled on the pod's cgroup.
	dmemFile = "dmem.max"

	// defaultCgroupRoot is the cgroup v2 filesystem mount on the host.
	defaultCgroupRoot = "/sys/fs/cgroup"
	// defaultProcRoot is the procfs mount (parameterized for tests).
	defaultProcRoot = "/proc"
)

// bdfPattern matches a full PCI BDF as sysfs reports it, e.g. "0000:e3:00.0".
var bdfPattern = regexp.MustCompile(`^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$`)

// podSlicePattern matches the pod cgroup under systemd's cgroup driver, e.g.
// "containers-kubepods-burstable-pod<uid>.slice". The container's own
// cri-containerd-*.scope lives inside it and charges to it, so the dmem cap
// must be written on this ancestor. Anchored on the pod-UID shape (at least 6
// hex/underscore/dash chars after "pod") so a stray "foo-podbar.slice"
// workload slice cannot be mistaken for a pod cgroup.
var podSlicePattern = regexp.MustCompile(`^.+-pod[0-9a-f][0-9a-f_-]{5,}\.slice$`)

// ociState is the subset of the OCI runtime state JSON (passed on stdin to
// createRuntime hooks) that we need.
type ociState struct {
	ID          string            `json:"id"`
	PID         int               `json:"pid"`
	Bundle      string            `json:"bundle"`
	Annotations map[string]string `json:"annotations"`
}

// ociConfig is the subset of the bundle's config.json we need.
type ociConfig struct {
	Linux struct {
		CgroupsPath string `json:"cgroupsPath"`
	} `json:"linux"`
}

func main() {
	done, err := handleFlags(os.Args[1:], os.Stdout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dmem-hook: ERROR: %v\n%s", err, usageText)
		os.Exit(2)
	}
	if done {
		return
	}
	if err := run(os.Stdin, os.Stdout, os.Stderr, defaultCgroupRoot, defaultProcRoot); err != nil {
		fmt.Fprintf(os.Stderr, "dmem-hook: ERROR: %v\n", err)
		os.Exit(1)
	}
}

// handleFlags implements --version/--usage. Returns done=true if the process
// handled its business; an error means an unknown argument.
func handleFlags(args []string, stdout io.Writer) (bool, error) {
	for _, arg := range args {
		switch arg {
		case "--version", "-v":
			fmt.Fprintf(stdout, "dmem-hook %s\n", version)
			return true, nil
		case "--usage", "--help", "-h":
			fmt.Fprint(stdout, usageText)
			return true, nil
		default:
			return false, fmt.Errorf("unknown argument %q", arg)
		}
	}
	return false, nil
}

const usageText = `dmem-hook - OCI createRuntime hook writing a kernel dmem VRAM cap on the pod cgroup

Usage: dmem-hook [--version | --usage]

Reads the OCI runtime state JSON from stdin and writes one line to dmem.max on
the pod's cgroup slice:

	drm/<AMD_GPU_PCI_BDF>/vram <AMD_GPU_MEM_BYTES>

Environment:
	AMD_GPU_PCI_BDF    PCI BDF of the allocated GPU (e.g. 0000:e3:00.0)
	AMD_GPU_MEM_BYTES  VRAM budget in bytes for this pod on that GPU
`

func run(stdin io.Reader, stdout, stderr io.Writer, cgroupRoot, procRoot string) error {
	var state ociState
	if err := json.NewDecoder(stdin).Decode(&state); err != nil {
		return fmt.Errorf("parsing OCI state from stdin: %w", err)
	}

	// Region names in the kernel's dmem controller are lowercase (sysfs-style
	// BDFs); normalize so a mixed-case env cannot target a nonexistent region.
	bdf := strings.ToLower(os.Getenv(envPCIBDF))
	if !bdfPattern.MatchString(bdf) {
		return fmt.Errorf("env %s=%q is not a full PCI BDF (want e.g. 0000:e3:00.0)", envPCIBDF, bdf)
	}
	memStr := os.Getenv(envMemBytes)
	memBytes, err := strconv.ParseUint(memStr, 10, 64)
	if err != nil || memBytes == 0 {
		return fmt.Errorf("env %s=%q is not a positive byte count: %w", envMemBytes, memStr, err)
	}

	// Attribute every message to the container we were invoked for.
	who := fmt.Sprintf("container %q (pid %d)", state.ID, state.PID)

	cgroupDir, source, err := resolveCgroupDir(state, cgroupRoot, procRoot, stderr)
	if err != nil {
		return fmt.Errorf("%s: resolving container cgroup: %w", who, err)
	}
	podSlice, err := findPodSlice(cgroupDir)
	if err != nil {
		return fmt.Errorf("%s: cgroup %q (from %s): %w", who, cgroupDir, source, err)
	}

	// Content format is the kernel's dmem.max per-region line. One region per
	// write; rewriting the same region replaces its entry (idempotent).
	line := fmt.Sprintf("drm/%s/vram %d\n", bdf, memBytes)
	target := filepath.Join(podSlice, dmemFile)
	if err := os.WriteFile(target, []byte(line), 0o644); err != nil {
		return fmt.Errorf("%s: writing %q to %s: %w", who, strings.TrimSpace(line), target, err)
	}

	// Read the entry back and assert it is there: if the kernel's dmem.max
	// format or region naming differs from our assumption, the write would
	// otherwise be a silent no-op — the exact failure this hook must not have.
	back, err := os.ReadFile(target)
	if err != nil {
		return fmt.Errorf("%s: reading back %s after write: %w", who, target, err)
	}
	if !strings.Contains(string(back), strings.TrimSpace(line)) {
		return fmt.Errorf("%s: %s does not contain %q after write (got %q); kernel dmem region naming or file format mismatch",
			who, target, strings.TrimSpace(line), strings.TrimSpace(string(back)))
	}

	fmt.Fprintf(stdout, "dmem-hook: %s: capped %s to %d bytes via %s (cgroup from %s)\n",
		who, strings.TrimSpace(line), memBytes, target, source)
	return nil
}

// resolveCgroupDir determines the container's cgroup directory. The bundle's
// config.json linux.cgroupsPath is preferred (it is what the runtime actually
// used); procRoot/<pid>/cgroup is the fallback when the bundle is unreadable or
// the path cannot be resolved. cgroupRoot is the cgroup v2 mount
// ("/sys/fs/cgroup" on the host) and procRoot is procfs ("/proc"); both are
// parameterized for tests.
//
// It returns the directory and which source resolved it (for log attribution).
func resolveCgroupDir(state ociState, cgroupRoot, procRoot string, stderr io.Writer) (string, string, error) {
	var errs []error

	if state.Bundle != "" {
		cfgPath := filepath.Join(state.Bundle, "config.json")
		raw, err := readCgroupsPath(cfgPath)
		if err != nil {
			errs = append(errs, fmt.Errorf("config.json: %w", err))
		} else if raw == "" {
			errs = append(errs, fmt.Errorf("config.json: linux.cgroupsPath is empty"))
		} else if dir, err := cgroupDirFromPath(raw, cgroupRoot); err != nil {
			errs = append(errs, fmt.Errorf("config.json cgroupsPath %q: %w", raw, err))
		} else {
			return dir, "config.json linux.cgroupsPath", nil
		}
	} else {
		errs = append(errs, errors.New("config.json: OCI state has no bundle"))
	}

	if state.PID > 0 {
		procCgroupPath := filepath.Join(procRoot, strconv.Itoa(state.PID), "cgroup")
		raw, err := readProcCgroup(procCgroupPath)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", procCgroupPath, err))
		} else if dir, err := cgroupDirFromPath(raw, cgroupRoot); err != nil {
			errs = append(errs, fmt.Errorf("proc cgroup path %q: %w", raw, err))
		} else {
			fmt.Fprintln(stderr, "dmem-hook: WARNING: falling back to /proc/<pid>/cgroup for cgroup resolution")
			return dir, "/proc/<pid>/cgroup", nil
		}
	} else {
		errs = append(errs, errors.New("/proc: OCI state has no pid"))
	}

	return "", "", errors.Join(errs...)
}

// readCgroupsPath returns linux.cgroupsPath from a bundle's config.json.
func readCgroupsPath(configPath string) (string, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return "", err
	}
	var cfg ociConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", fmt.Errorf("parsing %s: %w", configPath, err)
	}
	return cfg.Linux.CgroupsPath, nil
}

// readProcCgroup returns the cgroup v2 path of a process from its /proc entry.
func readProcCgroup(procCgroupPath string) (string, error) {
	data, err := os.ReadFile(procCgroupPath)
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		// cgroup v2 unified hierarchy line: "0::<path>".
		if rest, ok := strings.CutPrefix(line, "0::"); ok {
			return rest, nil
		}
	}
	return "", errors.New("no cgroup v2 (0::) line; dmem caps require cgroup v2")
}

// cgroupDirFromPath turns a raw cgroupsPath into a directory on the host's
// cgroup v2 filesystem and checks it exists. It understands both forms the
// systemd cgroup driver produces and plain cgroupfs paths:
//
//   - "slice:prefix:name" (systemd triple, e.g.
//     "containers-kubepods-burstable-pod<uid>.slice:cri-containerd:<id>") is
//     flattened to "<slice>/<prefix>-<name>.scope", expanding the flattened
//     slice name into its nested directory chain the way systemd creates it;
//   - anything else is a cgroupfs path: absolute paths are tried as-is first
//     (a full /sys/fs/cgroup/... path, or an absolute path under a test
//     root), then under cgroupRoot; relative paths are joined under
//     cgroupRoot.
func cgroupDirFromPath(raw, cgroupRoot string) (string, error) {
	var dir string
	if strings.Count(raw, ":") == 2 {
		// systemd triple "slice:prefix:name" -> "<expanded slice>/<prefix>-<name>.scope".
		parts := strings.SplitN(raw, ":", 3)
		if parts[1] == "" || parts[2] == "" {
			return "", fmt.Errorf("systemd cgroup path %q has an empty prefix or name", raw)
		}
		var sliceDir string
		for _, comp := range strings.Split(parts[0], "/") {
			expanded, err := expandSlice(comp)
			if err != nil {
				return "", err
			}
			sliceDir += expanded
		}
		dir = filepath.Join(cgroupRoot, sliceDir, fmt.Sprintf("%s-%s.scope", parts[1], parts[2]))
	} else {
		if i := strings.Index(raw, ":"); i >= 0 {
			return "", fmt.Errorf("unsupported cgroup path form %q", raw)
		}
		if filepath.IsAbs(raw) {
			if fi, err := os.Stat(raw); err == nil && fi.IsDir() {
				dir = raw
			} else {
				dir = filepath.Join(cgroupRoot, raw)
			}
		} else {
			dir = filepath.Join(cgroupRoot, raw)
		}
	}
	fi, err := os.Stat(dir)
	if err != nil {
		return "", fmt.Errorf("cgroup dir %s does not exist: %w", dir, err)
	}
	if !fi.IsDir() {
		return "", fmt.Errorf("cgroup path %s is not a directory", dir)
	}
	return dir, nil
}

// expandSlice converts a flattened systemd slice name (e.g.
// "a-b-c.slice") into its nested cgroupfs path ("/a.slice/a-b.slice/a-b-c.slice"),
// matching how systemd materializes slices on cgroupfs (and runc's
// expandSlice). The root slice ("/" or "-") expands to "".
func expandSlice(slice string) (string, error) {
	const suffix = ".slice"
	if slice == "" || slice == "-" || slice == "/" {
		return "", nil
	}
	if !strings.HasSuffix(slice, suffix) {
		return "", fmt.Errorf("invalid slice name %q: missing %q suffix", slice, suffix)
	}
	name := strings.TrimSuffix(slice, suffix)
	parts := strings.Split(name, "-")
	var b strings.Builder
	for i := range parts {
		if parts[i] == "" {
			return "", fmt.Errorf("invalid slice name %q: empty dash-separated component", slice)
		}
		b.WriteByte('/')
		b.WriteString(strings.Join(parts[:i+1], "-"))
		b.WriteString(suffix)
	}
	return b.String(), nil
}

// findPodSlice walks up from the container's cgroup directory to the ancestor
// whose name matches the pod-slice pattern — the pod cgroup. The container's
// own scope (e.g. cri-containerd-<id>.scope) charges to the pod cgroup, so the
// dmem cap must be written there, not on the container's scope.
func findPodSlice(dir string) (string, error) {
	for cur := filepath.Clean(dir); ; {
		name := filepath.Base(cur)
		if podSlicePattern.MatchString(name) {
			return cur, nil
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return "", fmt.Errorf("no *-pod<uid>.slice ancestor found above %s; pod cgroups are expected to be systemd *-pod<uid>.slice units (kubelet cgroupDriver=systemd)", dir)
		}
		cur = parent
	}
}
