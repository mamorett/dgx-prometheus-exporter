package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// errSmiUnavailable means nvidia-smi could not be executed at all (missing
// binary, missing ELF interpreter, or a failed exec). It is distinct from
// nvidia-smi running and exiting non-zero, which still yields usable output.
var errSmiUnavailable = errors.New("nvidia-smi unavailable")

// smiFunc runs nvidia-smi. It is a field on collector so tests can drive the
// unavailable/degraded paths without touching the host.
type smiFunc func(args ...string) (string, error)

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
//
// A nil *float64 means the value is unknown and renders as nan: the series is
// always present so that a lost metric is visible as nan rather than as a
// series that silently disappears from queries and dashboards. memOK gates the
// four memory metrics, which come from a source independent of nvidia-smi.
type snapshot struct {
	memOK                       bool
	total, avail, used, gpuUsed int64
	procs                       []process
	util                        *float64
	memUtil                     *float64
	temp                        *float64
	power                       *float64
	appCount                    int
	info                        gpuInfo
	errs                        []string
}

// ok reports whether every source produced data. It drives dgx_collect_success.
func (s snapshot) ok() bool { return len(s.errs) == 0 }

// collector bundles everything needed to run one collection cycle against the
// host. The meminfo path and the nvidia-smi runner are overridable so tests can
// inject fixtures and failure modes.
type collector struct {
	meminfoPath string
	runSmi      smiFunc

	// lastErrs is the error set from the previous cycle. main.go compares it
	// against the current one so a persistently broken driver is logged once
	// on the transition instead of on every interval.
	lastErrs []string
}

func newCollector(meminfoPath string) *collector {
	return newCollectorWithSmi(meminfoPath, runNvidiaSmi)
}

func newCollectorWithSmi(meminfoPath string, run smiFunc) *collector {
	if meminfoPath == "" {
		meminfoPath = "/proc/meminfo"
	}
	if run == nil {
		run = runNvidiaSmi
	}
	return &collector{meminfoPath: meminfoPath, runSmi: run}
}

var (
	// | N/A   30C    P8      4W /  N/A  |  (temperature + power summary row)
	// Fallback only: used when the structured --query-gpu telemetry query fails.
	reTempPower = regexp.MustCompile(`\|\s*N/A\s+(\d+)C\s+\S+\s+([\d.]+)W\s*/\s*\S+\s*\|`)
	// | Not Supported          |      0%      Default |  (GPU-Util)
	// Fallback only. Note this piggybacks on the GB10's "Not Supported"
	// memory-usage cell, which is why the structured query is preferred.
	reUtil = regexp.MustCompile(`Not Supported\s*\|\s+(\d+)%`)
	// |    0   N/A   N/A            2408      G   /usr/lib/xorg/Xorg      18MiB |
	// Graphics (G) context rows from the human-readable table. There is no
	// query interface for graphics contexts, so the table is the only source.
	reGraphicsRow = regexp.MustCompile(`^\|\s+0\s+N/A\s+N/A\s+(\d+)\s+G\s+(.*?)\s+(\d+)(MiB|GiB)\s*\|$`)
	// | NVIDIA-SMI 580.173.02   Driver Version: 580.173.02   CUDA Version: 13.0 |
	// CUDA version is only in the banner; it is not a valid --query-gpu field.
	reCudaVersion = regexp.MustCompile(`CUDA Version:\s*([0-9][0-9.]*)`)
	reDigits      = regexp.MustCompile(`[0-9]+`)
)

// telemetryFields / infoFields are the structured --query-gpu field lists. A
// structured query is used instead of scraping the human-readable table because
// the table truncates wide columns and reflows with terminal width.
const (
	telemetryFields = "temperature.gpu,power.draw,utilization.gpu,utilization.memory"
	infoFields      = "name,driver_version,uuid,compute_mode,persistence_mode,pstate,pci.bus_id,vbios_version,compute_cap"
)

// readMeminfo reads the meminfo file and returns (total, avail) in bytes.
func readMeminfo(path string) (int64, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, fmt.Errorf("open %s: %w", path, err)
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
		n, err := strconv.ParseInt(num, 10, 64)
		if err != nil {
			continue
		}
		switch key {
		case "MemTotal":
			total = n * 1024
		case "MemAvailable":
			avail = n * 1024
		}
	}
	if err := sc.Err(); err != nil {
		return 0, 0, fmt.Errorf("read %s: %w", path, err)
	}
	if total == 0 {
		return 0, 0, fmt.Errorf("parse %s: no MemTotal", path)
	}
	return total, avail, nil
}

// runNvidiaSmi runs nvidia-smi with the given arguments and returns its stdout.
// A non-zero exit code does NOT produce an error: nvidia-smi often still prints
// a usable table on stdout, so whatever was produced is returned as data.
// errSmiUnavailable is returned only when the binary could not be executed.
func runNvidiaSmi(args ...string) (string, error) {
	cmd := exec.Command("nvidia-smi", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		var execErr *exec.Error
		if errors.As(err, &execErr) {
			// Missing binary or failed exec (e.g. no ELF interpreter inside a
			// scratch container). Nothing usable was produced.
			return "", fmt.Errorf("%w: %v: %s", errSmiUnavailable, execErr, strings.TrimSpace(stderr.String()))
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			// Ran, exited non-zero: still parse stdout.
			return stdout.String(), nil
		}
		return stdout.String(), err
	}
	return stdout.String(), nil
}

// smiCSV runs a --query-gpu query and returns the first GPU's trimmed fields.
// Only GPU 0 is read (see the Single GPU note in the README).
func (c *collector) smiCSV(fields string) ([]string, error) {
	out, err := c.runSmi("--query-gpu="+fields, "--format=csv,noheader,nounits")
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(out) == "" {
		return nil, errors.New("empty query result")
	}
	r := csv.NewReader(strings.NewReader(out))
	r.FieldsPerRecord = -1
	rec, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("parse query result %q: %w", out, err)
	}
	for i := range rec {
		rec[i] = strings.TrimSpace(rec[i])
	}
	return rec, nil
}

// parseSmiFloat converts an nvidia-smi scalar field to a float. N/A (which
// nvidia-smi spells "N/A" or "[N/A]" depending on the field) and blanks are
// unknown, not zero.
func parseSmiFloat(s string) *float64 {
	s = strings.TrimSpace(strings.Trim(s, "[]"))
	if s == "" || strings.EqualFold(s, "n/a") || strings.EqualFold(s, "not supported") {
		return nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	return &f
}

// parseTelemetry reads the structured telemetry row: temperature.gpu,
// power.draw, utilization.gpu, utilization.memory. Utilisation percentages are
// converted to ratios.
func parseTelemetry(fields []string) (temp, power, util, memUtil *float64) {
	if len(fields) < 4 {
		return nil, nil, nil, nil
	}
	temp = parseSmiFloat(fields[0])
	power = parseSmiFloat(fields[1])
	if u := parseSmiFloat(fields[2]); u != nil {
		r := *u / 100.0
		util = &r
	}
	if m := parseSmiFloat(fields[3]); m != nil {
		r := *m / 100.0
		memUtil = &r
	}
	return
}

// parseSmiTable parses the human-readable nvidia-smi table. It yields the
// graphics (G) contexts — available from no other source — the CUDA version
// from the banner, and temp/power/util as a fallback when the structured query
// is unavailable. appCount is always 0 here; it comes from the compute-apps CSV.
func parseSmiTable(out string) (gpuUsed int64, procs []process, util, memUtil, temp, power *float64, appCount int, cuda string) {
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
		if cuda == "" {
			if m := reCudaVersion.FindStringSubmatch(line); m != nil {
				cuda = m[1]
			}
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

// gatherGPUInfo builds the static GPU attributes for dgx_gpu_info. bannerCuda is
// the CUDA version scraped from the nvidia-smi banner by the caller, which
// already had the table output; it is not a valid --query-gpu field.
//
// The telemetry and info queries are deliberately kept as separate nvidia-smi
// invocations: folding them together would let one unrecognised info field
// fail the whole query and take the crucial temperature reading down with it.
func (c *collector) gatherGPUInfo(bannerCuda string) gpuInfo {
	info := gpuInfo{
		name:            "unknown",
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

	if bannerCuda != "" {
		info.cuda = bannerCuda
	}

	fields, err := c.smiCSV(infoFields)
	if err == nil && len(fields) >= 9 {
		info.name = fields[0]
		info.driver = fields[1]
		info.uuid = fields[2]
		info.computeMode = fields[3]
		info.persistenceMode = fields[4]
		info.pstate = fields[5]
		info.pciBusID = fields[6]
		info.vbios = fields[7]
		info.computeCap = fields[8]
	}

	host, _ := os.Hostname()
	if envHost := os.Getenv("DGX_HOST_NAME"); envHost != "" {
		host = envHost
	}
	info.host = host

	// Environment overrides are authoritative: inside a container the driver
	// reports what the injected compatibility library advertises, which need
	// not match the host driver.
	if v := os.Getenv("DGX_DRIVER_VERSION"); v != "" {
		info.driver = v
	}
	if v := os.Getenv("DGX_CUDA_VERSION"); v != "" {
		info.cuda = v
	}
	return info
}

// collect runs a full collection cycle and renders the Prometheus text.
//
// Partial failure is reported, not hidden: whatever sources did answer are still
// rendered, the unknowns render as nan, and dgx_collect_success goes to 0. An
// error is returned only when nothing at all could be collected, so that a
// degraded exporter keeps serving data instead of throwing it away.
func (c *collector) collect() (string, error) {
	s := snapshot{}

	total, avail, memErr := readMeminfo(c.meminfoPath)
	if memErr == nil {
		s.memOK = true
		s.total, s.avail = total, avail
		s.used = total - avail
	}

	smiOut, smiErr := c.runSmi()
	tblGPUUsed, tblProcs, tblUtil, _, tblTemp, tblPower, _, bannerCuda := parseSmiTable(smiOut)
	s.info = c.gatherGPUInfo(bannerCuda)

	// Structured query first (exact, width-independent); table regex as fallback.
	temp, power, util, memUtil := (*float64)(nil), (*float64)(nil), (*float64)(nil), (*float64)(nil)
	if fields, err := c.smiCSV(telemetryFields); err == nil {
		temp, power, util, memUtil = parseTelemetry(fields)
	}
	if temp == nil {
		temp = tblTemp
	}
	if power == nil {
		power = tblPower
	}
	if util == nil {
		util = tblUtil
	}

	s.temp, s.power, s.util, s.memUtil = temp, power, util, memUtil
	s.gpuUsed = tblGPUUsed
	s.procs = tblProcs

	if smiErr == nil {
		cappsOut, _ := c.runSmi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
		cProcs, appCount := parseComputeApps(cappsOut)
		for _, p := range cProcs {
			s.gpuUsed += p.bytes
		}
		s.procs = append(s.procs, cProcs...)
		s.appCount = appCount
	}

	// Record what failed. Each distinct loss is named so the served payload says
	// why it degraded.
	if memErr != nil {
		s.errs = append(s.errs, memErr.Error())
	}
	if smiErr != nil {
		s.errs = append(s.errs, smiErr.Error())
	} else {
		if s.temp == nil {
			s.errs = append(s.errs, "GPU temperature unavailable")
		}
		if s.power == nil {
			s.errs = append(s.errs, "GPU power draw unavailable")
		}
	}

	c.lastErrs = s.errs

	// Nothing at all came back: serve the error payload instead of an all-nan page.
	if !s.memOK && smiErr != nil {
		return "", fmt.Errorf("no metrics available: %w", errors.Join(append([]error{memErr}, smiErr)...))
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

// formatRatio renders a ratio, or nan when the source did not answer.
func formatRatio(r *float64) string {
	if r == nil {
		return "nan"
	}
	return strconv.FormatFloat(*r, 'f', 6, 64)
}

// formatTemp and formatPower exist so the crucial thermal metrics keep a stable
// rendering and always emit a line, nan included.
func formatTemp(v *float64) string {
	if v == nil {
		return "nan"
	}
	return strconv.FormatFloat(*v, 'f', 1, 64)
}

func formatPower(v *float64) string {
	if v == nil {
		return "nan"
	}
	return strconv.FormatFloat(*v, 'f', 2, 64)
}

func formatCount(v int64, ok bool) string {
	if !ok {
		return "nan"
	}
	return strconv.FormatInt(v, 10)
}

func render(s snapshot) string {
	var buf strings.Builder

	emitGauge(&buf, "dgx_unified_memory_total_bytes",
		"Total unified memory (host system RAM) available to the GPU.",
		formatCount(s.total, s.memOK), nil)
	emitGauge(&buf, "dgx_unified_memory_used_bytes",
		"Unified memory in active use (total - available).",
		formatCount(s.used, s.memOK), nil)
	emitGauge(&buf, "dgx_unified_memory_available_bytes",
		"Unified memory available for new allocations.",
		formatCount(s.avail, s.memOK), nil)
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

	// Always emitted: an unanswered field is nan, never an absent series.
	emitGauge(&buf, "dgx_gpu_memory_controller_ratio",
		"Memory controller utilization ratio.",
		formatRatio(s.memUtil), nil)
	emitGauge(&buf, "dgx_gpu_temperature_celsius",
		"GPU temperature in Celsius. nan only when nvidia-smi could not answer.",
		formatTemp(s.temp), nil)
	emitGauge(&buf, "dgx_gpu_power_draw_watts",
		"GPU power draw in watts. nan only when nvidia-smi could not answer.",
		formatPower(s.power), nil)

	emitGauge(&buf, "dgx_gpu_compute_apps",
		"Number of processes with a compute context on the GPU.",
		strconv.Itoa(s.appCount), nil)

	infoLabels := []label{
		{"name", s.info.name},
		{"driver", s.info.driver},
		{"cuda", s.info.cuda},
		{"uuid", s.info.uuid},
		{"pci_bus_id", s.info.pciBusID},
		{"host", s.info.host},
		{"vbios", s.info.vbios},
		{"compute_cap", s.info.computeCap},
		{"pstate", s.info.pstate},
		{"compute_mode", s.info.computeMode},
	}
	emitGauge(&buf, "dgx_gpu_info", "Static GPU info.", "1", infoLabels)
	emitGauge(&buf, "dgx_spark_info", "DGX Spark host info.", "1", []label{{"host", s.info.host}})

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

	success := 1
	if !s.ok() {
		success = 0
	}
	emitGauge(&buf, "dgx_collect_success",
		"Whether collection succeeded (1) or not (0).",
		strconv.Itoa(success), nil)

	for _, e := range s.errs {
		buf.WriteString("# collector error: " + e + "\n")
	}

	return buf.String()
}

// errorPayload is the text served when a collection produced no data at all.
func errorPayload(err error) string {
	return "# HELP dgx_collect_success Whether collection succeeded (1) or not (0).\n" +
		"# TYPE dgx_collect_success gauge\n" +
		"dgx_collect_success 0\n" +
		"# collector error: " + err.Error() + "\n"
}
