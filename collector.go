package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// process describes one GPU context (graphics G or compute C).
type process struct {
	pid   string
	kind  string // "G" graphics or "C" compute
	name  string
	bytes int64
}

// gpuInfo holds the static GPU attributes that are reported in dgx_gpu_info.
type gpuInfo struct {
	name, driver, cuda, uuid, computeMode, persistenceMode string
	pstate, pciBusID, vbios, computeCap, host              string
}

// snapshot is a single collection cycle's worth of data, rendered into the
// Prometheus text format by render.
type snapshot struct {
	total, avail, used, gpuUsed int64
	procs                       []process
	util                        *float64 // nil = N/A
	memUtil                     *float64
	temp                        *float64
	power                       *float64
	appCount                    int
	info                        gpuInfo
}

// collector bundles everything needed to run one collection cycle against the
// host. The meminfo path is overridable so tests can inject a fixture.
type collector struct {
	meminfoPath string
}

func newCollector(meminfoPath string) *collector {
	if meminfoPath == "" {
		meminfoPath = "/proc/meminfo"
	}
	return &collector{meminfoPath: meminfoPath}
}

var (
	// | N/A   30C    P8      4W /  N/A  |  (temperature + power summary row)
	reTempPower = regexp.MustCompile(`\|\s*N/A\s+(\d+)C\s+\S+\s+([\d.]+)W\s*/\s*\S+\s*\|`)
	// | Not Supported          |      0%      Default |  (GPU-Util)
	reUtil = regexp.MustCompile(`Not Supported\s*\|\s+(\d+)%`)
	// |    0   N/A   N/A            2408      G   /usr/lib/xorg/Xorg      18MiB |
	// Graphics (G) context rows from the human-readable table.
	reGraphicsRow = regexp.MustCompile(`^\|\s+0\s+N/A\s+N/A\s+(\d+)\s+G\s+(.*?)\s+(\d+)(MiB|GiB)\s*\|$`)
	reDigits      = regexp.MustCompile(`[0-9]+`)
)

// readMeminfo reads the meminfo file and returns (total, avail) in bytes.
// It returns (0, 0) when the file is missing or unreadable.
func readMeminfo(path string) (int64, int64) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0
	}
	defer f.Close()

	total, avail := int64(0), int64(0)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		key, val, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		num := reDigits.FindString(val)
		if num == "" {
			continue
		}
		n, _ := strconv.Atoi(num)
		switch key {
		case "MemTotal":
			total = int64(n) * 1024
		case "MemAvailable":
			avail = int64(n) * 1024
		}
	}
	return total, avail
}

// runNvidiaSmi runs nvidia-smi with the given arguments and returns its stdout.
// When the nvidia-smi binary is missing, it returns ("", nil) so the caller
// treats it the same as "no data". A non-zero exit code does NOT cause an
// error to be returned: whatever was produced on stdout is still valid data.
func runNvidiaSmi(args ...string) (string, error) {
	cmd := exec.Command("nvidia-smi", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		var execErr *exec.Error
		if errors.As(err, &execErr) {
			// Binary not found -> no data, not an error.
			return "", nil
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			// Ran, exited non-zero: still parse stdout.
			return stdout.String(), nil
		}
		// Unexpected error: return whatever we managed to capture.
		return stdout.String(), err
	}
	return stdout.String(), nil
}

// parseSmiTable parses the human-readable nvidia-smi table.
// It returns the GPU usage and the list of per-process graphics (G) contexts,
// plus the summary telemetry (util, memUtil, temp, power).
// appCount is always 0 here — it is filled in by the compute-apps CSV step.
func parseSmiTable(out string) (gpuUsed int64, procs []process, util, memUtil, temp, power *float64, appCount int) {
	for _, line := range strings.Split(out, "\n") {
		if m := reTempPower.FindStringSubmatch(line); m != nil {
			t, _ := strconv.ParseFloat(m[1], 64)
			p, _ := strconv.ParseFloat(m[2], 64)
			temp = &t
			power = &p
		}
		if m := reUtil.FindStringSubmatch(line); m != nil {
			u, _ := strconv.Atoi(m[1])
			f := float64(u) / 100.0
			util = &f
		}
		if m := reGraphicsRow.FindStringSubmatch(line); m != nil {
			pid := m[1]
			name := strings.TrimSpace(m[2])
			amount, _ := strconv.Atoi(m[3])
			if m[4] == "GiB" {
				amount *= 1024
			}
			b := int64(amount) * 1024 * 1024
			gpuUsed += b
			procs = append(procs, process{pid: pid, kind: "G", name: name, bytes: b})
		}
	}
	return
}

// parseComputeApps parses the CSV output of
// nvidia-smi --query-compute-apps=pid,process_name,used_memory
// --format=csv,noheader,nounits
// and returns the list of compute (C) contexts plus the number of rows
// successfully parsed.
func parseComputeApps(out string) ([]process, int) {
	// Drop blank lines (Python: `[ln for ln in capps.splitlines() if ln.strip()]`).
	lines := make([]string, 0, 16)
	for _, ln := range strings.Split(out, "\n") {
		if strings.TrimSpace(ln) != "" {
			lines = append(lines, ln)
		}
	}
	if len(lines) == 0 {
		return nil, 0
	}

	var procs []process
	count := 0
	r := csv.NewReader(strings.NewReader(strings.Join(lines, "\n")))
	r.FieldsPerRecord = -1
	for {
		rec, err := r.Read()
		if err != nil {
			break
		}
		if len(rec) < 3 {
			continue
		}
		pid := strings.TrimSpace(rec[0])
		name := strings.TrimSpace(rec[1])
		mib, err := strconv.ParseInt(strings.TrimSpace(rec[2]), 10, 64)
		if err != nil {
			continue // skip malformed rows (Python catches ValueError)
		}
		b := mib * 1024 * 1024
		procs = append(procs, process{pid: pid, kind: "C", name: name, bytes: b})
		count++
	}
	return procs, count
}

// gatherGPUInfo builds the static GPU info, mirroring the Python _nvidia_driver.
func (c *collector) gatherGPUInfo() gpuInfo {
	info := gpuInfo{
		name:            "NVIDIA GB10",
		driver:          "unknown",
		cuda:            "unknown",
		uuid:            "unknown",
		computeMode:     "unknown",
		persistenceMode: "unknown",
		pstate:          "unknown",
		pciBusID:        "unknown",
		vbios:           "unknown",
		computeCap:      "unknown",
		host:            "unknown",
	}

	out, _ := runNvidiaSmi(
		"--query-gpu=name,driver_version,uuid,compute_mode,persistence_mode,pstate,pci.bus_id,vbios_version,compute_cap",
		"--format=csv,noheader,nounits",
	)
	parts := strings.Split(out, ",")
	if len(parts) >= 9 {
		info.name = strings.TrimSpace(parts[0])
		info.driver = strings.TrimSpace(parts[1])
		info.uuid = strings.TrimSpace(parts[2])
		info.computeMode = strings.TrimSpace(parts[3])
		info.persistenceMode = strings.TrimSpace(parts[4])
		info.pstate = strings.TrimSpace(parts[5])
		info.pciBusID = strings.TrimSpace(parts[6])
		info.vbios = strings.TrimSpace(parts[7])
		info.computeCap = strings.TrimSpace(parts[8])
	}

	host, _ := os.Hostname()
	if envHost := os.Getenv("DGX_HOST_NAME"); envHost != "" {
		host = envHost
	}
	info.host = host

	if v := os.Getenv("DGX_DRIVER_VERSION"); v != "" {
		info.driver = v
	}
	if v := os.Getenv("DGX_CUDA_VERSION"); v != "" {
		info.cuda = v
	}
	return info
}

// collect runs a full collection cycle and renders the Prometheus text.
func (c *collector) collect() (string, error) {
	total, avail := readMeminfo(c.meminfoPath)
	used := total - avail

	smiOut, _ := runNvidiaSmi()
	gpuUsed, procs, util, memUtil, temp, power, _ := parseSmiTable(smiOut)
	cappsOut, _ := runNvidiaSmi(
		"--query-compute-apps=pid,process_name,used_memory",
		"--format=csv,noheader,nounits",
	)
	cProcs, appCount := parseComputeApps(cappsOut)
	for _, p := range cProcs {
		gpuUsed += p.bytes
	}
	procs = append(procs, cProcs...)

	info := c.gatherGPUInfo()

	s := snapshot{
		total:    total,
		avail:    avail,
		used:     used,
		gpuUsed:  gpuUsed,
		procs:    procs,
		util:     util,
		memUtil:  memUtil,
		temp:     temp,
		power:    power,
		appCount: appCount,
		info:     info,
	}
	return render(s), nil
}

// label is one (name, value) pair for a metric series.
type label struct {
	k, v string
}

func escape(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	return s
}

func formatLabels(labels []label) string {
	if len(labels) == 0 {
		return ""
	}
	parts := make([]string, len(labels))
	for i, l := range labels {
		parts[i] = l.k + "=\"" + escape(l.v) + "\""
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func emitGauge(buf *strings.Builder, name, help, value string, labels []label) {
	lbl := formatLabels(labels)
	buf.WriteString("# HELP " + name + " " + help + "\n")
	buf.WriteString("# TYPE " + name + " gauge\n")
	buf.WriteString(name + lbl + " " + value + "\n")
}

func formatRatio(r *float64) string {
	if r == nil {
		return "nan"
	}
	return strconv.FormatFloat(*r, 'f', 6, 64)
}

func render(s snapshot) string {
	var buf strings.Builder

	emitGauge(&buf, "dgx_unified_memory_total_bytes",
		"Total unified memory (host system RAM) available to the GPU.",
		strconv.FormatInt(s.total, 10), nil)
	emitGauge(&buf, "dgx_unified_memory_used_bytes",
		"Unified memory in active use (total - available).",
		strconv.FormatInt(s.used, 10), nil)
	emitGauge(&buf, "dgx_unified_memory_available_bytes",
		"Unified memory available for new allocations.",
		strconv.FormatInt(s.avail, 10), nil)
	emitGauge(&buf, "dgx_unified_memory_gpu_used_bytes",
		"Unified memory held by active GPU processes (sum of compute+graphics).",
		strconv.FormatInt(s.gpuUsed, 10), nil)

	// Per-process series: one HELP/TYPE block, then one line per process.
	buf.WriteString("# HELP dgx_unified_memory_process_used_bytes Unified memory held by one GPU process (compute C or graphics G context).\n")
	buf.WriteString("# TYPE dgx_unified_memory_process_used_bytes gauge\n")
	for _, p := range s.procs {
		buf.WriteString("dgx_unified_memory_process_used_bytes{pid=\"" + escape(p.pid) +
			"\",type=\"" + p.kind +
			"\",process_name=\"" + escape(p.name) + "\"} " +
			strconv.FormatInt(p.bytes, 10) + "\n")
	}

	emitGauge(&buf, "dgx_gpu_utilization_ratio",
		"GPU compute utilization (nvidia-smi GPU-Util) as a ratio.",
		formatRatio(s.util), nil)

	if s.memUtil != nil {
		emitGauge(&buf, "dgx_gpu_memory_controller_ratio",
			"Memory controller utilization ratio.",
			formatRatio(s.memUtil), nil)
	}
	if s.temp != nil {
		emitGauge(&buf, "dgx_gpu_temperature_celsius",
			"GPU temperature in Celsius.",
			strconv.FormatFloat(*s.temp, 'f', 1, 64), nil)
	}
	if s.power != nil {
		emitGauge(&buf, "dgx_gpu_power_draw_watts",
			"GPU power draw in watts.",
			strconv.FormatFloat(*s.power, 'f', 2, 64), nil)
	}

	emitGauge(&buf, "dgx_gpu_compute_apps",
		"Number of processes with a compute context on the GPU.",
		strconv.Itoa(s.appCount), nil)

	uuid := s.info.uuid
	if len(uuid) > 8 {
		uuid = uuid[:8]
	}
	infoLabels := []label{
		{"name", s.info.name},
		{"driver", s.info.driver},
		{"cuda", s.info.cuda},
		{"uuid", uuid},
		{"pci_bus_id", s.info.pciBusID},
		{"host", s.info.host},
		{"vbios", s.info.vbios},
		{"compute_cap", s.info.computeCap},
		{"pstate", s.info.pstate},
		{"compute_mode", s.info.computeMode},
	}
	emitGauge(&buf, "dgx_gpu_info", "Static GPU info.", "1", infoLabels)

	emitGauge(&buf, "dgx_gpu_pstate", "GPU performance state (NVIDIA P-state).",
		"1", []label{{"pstate", s.info.pstate}})
	emitGauge(&buf, "dgx_gpu_compute_mode", "GPU compute mode.",
		"1", []label{{"mode", s.info.computeMode}})
	modeEnabled := 0
	if s.info.computeMode == "Exclusive_Process" {
		modeEnabled = 1
	}
	emitGauge(&buf, "dgx_gpu_compute_mode_enabled",
		"1 if compute mode is 'Exclusive_Process'.",
		strconv.Itoa(modeEnabled), nil)
	emitGauge(&buf, "dgx_gpu_persistence_mode", "GPU persistence mode.",
		"1", []label{{"mode", s.info.persistenceMode}})
	persistEnabled := 0
	if strings.HasPrefix(strings.ToLower(s.info.persistenceMode), "enabled") {
		persistEnabled = 1
	}
	emitGauge(&buf, "dgx_gpu_persistence_mode_enabled",
		"1 if persistence mode is Enabled (0 if Disabled).",
		strconv.Itoa(persistEnabled), nil)
	emitGauge(&buf, "dgx_collect_success",
		"Whether collection succeeded (1) or not (0).", "1", nil)

	return buf.String()
}

// errorPayload is the exact text served when a collection fails.
func errorPayload(err error) string {
	return "# HELP dgx_collect_success Whether collection succeeded (1) or not (0).\n" +
		"# TYPE dgx_collect_success gauge\n" +
		"dgx_collect_success 0\n" +
		"# collector error: " + err.Error() + "\n"
}
