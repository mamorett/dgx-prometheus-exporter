package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// fakeSmi is an injectable nvidia-smi. It routes on the argument shape the
// collector actually issues, so a test can make exactly one source fail.
type fakeSmi struct {
	table string
	tele  string
	info  string
	apps  string
	err   error // when set, every invocation fails with it
}

func (f fakeSmi) run(args ...string) (string, error) {
	if f.err != nil {
		return "", f.err
	}
	joined := strings.Join(args, " ")
	switch {
	case strings.HasPrefix(joined, "--query-gpu=temperature"):
		return f.tele, nil
	case strings.HasPrefix(joined, "--query-gpu=name"):
		return f.info, nil
	case strings.HasPrefix(joined, "--query-compute-apps"):
		return f.apps, nil
	default:
		return f.table, nil
	}
}

// Values below are the real shapes produced by driver 580.173.02 on a GB10.
const (
	fakeTele = "52, 9.43, 0, 0"
	fakeInfo = "NVIDIA GB10, 580.173.02, GPU-0cddbf68-70f0-0aa4-7f92-a624e48fef64, Default, Enabled, P0, 0000000F:01:00.0, 9A.0B.1E.00.00, 12.1"
	fakeApps = "212029, VLLM::EngineCore, 105530\n"
)

const fakeTable = `+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| N/A   48C    P0              9W /  N/A  | Not Supported          |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
|    0   N/A  N/A            3623      G   /usr/lib/xorg/Xorg                       47MiB |
+-----------------------------------------+----------------------+
`

// tableTempPower has the telemetry row but no structured-query counterpart.
const tableTempPower = `| NVIDIA-SMI 580.173.02   CUDA Version: 13.0     |
| N/A   48C    P0              9W /  N/A  | Not Supported          |      0%      Default |
`

// writeMeminfo drops a fixture meminfo into a temp dir and returns its path.
func writeMeminfo(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "meminfo")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

const meminfoFixture = `MemTotal:       128000000 kB
MemFree:         10000000 kB
MemAvailable:    90000000 kB
Buffers:            100000 kB
Cached:            2000000 kB
`

func TestReadMeminfo(t *testing.T) {
	path := writeMeminfo(t, meminfoFixture)
	total, avail, err := readMeminfo(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := int64(128000000) * 1024; total != want {
		t.Errorf("total = %d, want %d", total, want)
	}
	if want := int64(90000000) * 1024; avail != want {
		t.Errorf("avail = %d, want %d", avail, want)
	}
}

func TestReadMeminfoMissingIsAnError(t *testing.T) {
	total, avail, err := readMeminfo("/nonexistent/meminfo")
	if err == nil {
		t.Fatal("expected an error for an unreadable meminfo, got none")
	}
	if total != 0 || avail != 0 {
		t.Errorf("expected (0,0), got (%d,%d)", total, avail)
	}
}

func TestReadMeminfoNoMemTotalIsAnError(t *testing.T) {
	path := writeMeminfo(t, "MemFree: 1 kB\n")
	if _, _, err := readMeminfo(path); err == nil {
		t.Fatal("expected an error when MemTotal is absent")
	}
}

func TestParseSmiTable(t *testing.T) {
	gpuUsed, procs, util, memUtil, temp, power, appCount, cuda := parseSmiTable(fakeTable)

	wantGPUUsed := int64(47) * 1024 * 1024
	if gpuUsed != wantGPUUsed {
		t.Errorf("gpuUsed = %d, want %d", gpuUsed, wantGPUUsed)
	}
	if appCount != 0 {
		t.Errorf("appCount = %d, want 0 (table step does not count compute apps)", appCount)
	}
	if memUtil != nil {
		t.Errorf("memUtil = %v, want nil (the table has no memory-controller field)", *memUtil)
	}
	if util == nil || *util != 0.0 {
		t.Errorf("util = %v, want 0.0", util)
	}
	if temp == nil || *temp != 48.0 {
		t.Errorf("temp = %v, want 48.0", temp)
	}
	if power == nil || *power != 9.0 {
		t.Errorf("power = %v, want 9.0", power)
	}
	if cuda != "13.0" {
		t.Errorf("cuda = %q, want 13.0", cuda)
	}
	if len(procs) != 1 {
		t.Fatalf("len(procs) = %d, want 1", len(procs))
	}
	p := procs[0]
	if p.pid != "3623" || p.kind != "G" || p.name != "/usr/lib/xorg/Xorg" {
		t.Errorf("procs[0] = %+v", p)
	}
	if p.bytes != wantGPUUsed {
		t.Errorf("bytes = %d, want %d", p.bytes, wantGPUUsed)
	}
}

func TestParseSmiTableEmpty(t *testing.T) {
	gpuUsed, procs, util, memUtil, temp, power, appCount, cuda := parseSmiTable("")
	if gpuUsed != 0 || util != nil || memUtil != nil || temp != nil || power != nil || appCount != 0 || cuda != "" {
		t.Errorf("expected zero results, got gpuUsed=%d util=%v memUtil=%v temp=%v power=%v appCount=%d cuda=%q",
			gpuUsed, util, memUtil, temp, power, appCount, cuda)
	}
	if len(procs) != 0 {
		t.Errorf("len(procs) = %d, want 0", len(procs))
	}
}

func TestParseTelemetry(t *testing.T) {
	temp, power, util, memUtil := parseTelemetry([]string{"52", "9.43", "0", "0"})
	if temp == nil || *temp != 52 {
		t.Errorf("temp = %v, want 52", temp)
	}
	if power == nil || *power != 9.43 {
		t.Errorf("power = %v, want 9.43", power)
	}
	if util == nil || *util != 0 {
		t.Errorf("util = %v, want 0", util)
	}
	if memUtil == nil || *memUtil != 0 {
		t.Errorf("memUtil = %v, want 0", memUtil)
	}
}

func TestParseTelemetryRatiosArePercentScaled(t *testing.T) {
	_, _, util, memUtil := parseTelemetry([]string{"52", "9.43", "84", "12"})
	if util == nil || *util != 0.84 {
		t.Errorf("util = %v, want 0.84", util)
	}
	if memUtil == nil || *memUtil != 0.12 {
		t.Errorf("memUtil = %v, want 0.12", memUtil)
	}
}

func TestParseTelemetryShortRowIsAllUnknown(t *testing.T) {
	temp, power, util, memUtil := parseTelemetry([]string{"52", "9.43"})
	if temp != nil || power != nil || util != nil || memUtil != nil {
		t.Errorf("expected all nil for a short row, got %v %v %v %v", temp, power, util, memUtil)
	}
}

func TestParseSmiFloatNAForms(t *testing.T) {
	for _, s := range []string{"", " ", "N/A", "n/a", "[N/A]", "[Not Supported]", "Not Supported", "bogus"} {
		if got := parseSmiFloat(s); got != nil {
			t.Errorf("parseSmiFloat(%q) = %v, want nil", s, *got)
		}
	}
	for _, s := range []string{"0", " 52 ", "9.43", "[9.43]"} {
		if got := parseSmiFloat(s); got == nil {
			t.Errorf("parseSmiFloat(%q) = nil, want a value", s)
		}
	}
}

func TestParseComputeApps(t *testing.T) {
	out := "469190, VLLM::EngineCore, 111255\n12345, python, 128\nmalformed,row\n"
	procs, count := parseComputeApps(out)
	if count != 2 {
		t.Errorf("count = %d, want 2", count)
	}
	if len(procs) != 2 {
		t.Fatalf("len(procs) = %d, want 2", len(procs))
	}
	if procs[0].pid != "469190" || procs[0].kind != "C" || procs[0].name != "VLLM::EngineCore" {
		t.Errorf("procs[0] = %+v", procs[0])
	}
	if want := int64(111255) * 1024 * 1024; procs[0].bytes != want {
		t.Errorf("procs[0].bytes = %d, want %d", procs[0].bytes, want)
	}
	if procs[1].pid != "12345" || procs[1].name != "python" {
		t.Errorf("procs[1] = %+v", procs[1])
	}
}

func TestParseComputeAppsEmpty(t *testing.T) {
	procs, count := parseComputeApps("")
	if count != 0 || len(procs) != 0 {
		t.Errorf("expected (nil, 0), got (%d procs, %d count)", len(procs), count)
	}
}

func TestParseComputeAppsMalformedSkipped(t *testing.T) {
	procs, count := parseComputeApps("abc, python, not_a_number\n")
	if count != 0 || len(procs) != 0 {
		t.Errorf("expected nothing parsed, got procs=%v count=%d", procs, count)
	}
}

func TestGatherGPUInfoEnvOverrides(t *testing.T) {
	t.Setenv("DGX_HOST_NAME", "dgx1")
	t.Setenv("DGX_DRIVER_VERSION", "565.70")
	t.Setenv("DGX_CUDA_VERSION", "12.4")

	c := newCollectorWithSmi("/nonexistent/meminfo", fakeSmi{info: fakeInfo}.run)
	info := c.gatherGPUInfo("13.0")

	if info.host != "dgx1" {
		t.Errorf("host = %q, want dgx1", info.host)
	}
	if info.driver != "565.70" {
		t.Errorf("driver = %q, want 565.70 (env must win over nvidia-smi)", info.driver)
	}
	if info.cuda != "12.4" {
		t.Errorf("cuda = %q, want 12.4 (env must win over the banner)", info.cuda)
	}
}

func TestGatherGPUInfoFromQuery(t *testing.T) {
	t.Setenv("DGX_HOST_NAME", "dgx1")
	t.Setenv("DGX_DRIVER_VERSION", "")
	t.Setenv("DGX_CUDA_VERSION", "")

	c := newCollectorWithSmi("", fakeSmi{info: fakeInfo}.run)
	info := c.gatherGPUInfo("13.0")

	if info.name != "NVIDIA GB10" {
		t.Errorf("name = %q", info.name)
	}
	if info.driver != "580.173.02" {
		t.Errorf("driver = %q", info.driver)
	}
	if info.cuda != "13.0" {
		t.Errorf("cuda = %q, want 13.0 from the banner", info.cuda)
	}
	if info.uuid != "GPU-0cddbf68-70f0-0aa4-7f92-a624e48fef64" {
		t.Errorf("uuid = %q, want the full UUID untruncated", info.uuid)
	}
	if info.computeCap != "12.1" || info.vbios != "9A.0B.1E.00.00" || info.pstate != "P0" {
		t.Errorf("info = %+v", info)
	}
}

func TestGatherGPUInfoUnknownWhenQueryFails(t *testing.T) {
	t.Setenv("DGX_HOST_NAME", "")
	t.Setenv("DGX_DRIVER_VERSION", "")
	t.Setenv("DGX_CUDA_VERSION", "")

	c := newCollectorWithSmi("", fakeSmi{err: errSmiUnavailable}.run)
	info := c.gatherGPUInfo("")

	host, _ := os.Hostname()
	if info.host != host {
		t.Errorf("host = %q, want %q", info.host, host)
	}
	for name, v := range map[string]string{
		"name": info.name, "driver": info.driver, "cuda": info.cuda, "uuid": info.uuid,
		"computeCap": info.computeCap, "pstate": info.pstate,
	} {
		if v != "unknown" {
			t.Errorf("%s = %q, want \"unknown\"", name, v)
		}
	}
}

func TestSnapshotOK(t *testing.T) {
	if !(snapshot{}).ok() {
		t.Error("a snapshot with no errors must report ok")
	}
	if (snapshot{errs: []string{"boom"}}).ok() {
		t.Error("a snapshot with errors must not report ok")
	}
}

func TestRenderGolden(t *testing.T) {
	temp, power, util, memUtil := 52.0, 9.43, 0.0, 0.0
	gBytes := int64(18) * 1024 * 1024
	cBytes := int64(111255) * 1024 * 1024
	s := snapshot{
		memOK:   true,
		total:   100 * 1024 * 1024,
		avail:   50 * 1024 * 1024,
		used:    50 * 1024 * 1024,
		gpuUsed: gBytes + cBytes,
		procs: []process{
			{pid: "2408", kind: "G", name: "/usr/lib/xorg/Xorg", bytes: gBytes},
			{pid: "469190", kind: "C", name: "VLLM::EngineCore", bytes: cBytes},
		},
		util:     &util,
		memUtil:  &memUtil,
		temp:     &temp,
		power:    &power,
		appCount: 1,
		info: gpuInfo{
			name: "NVIDIA GB10", driver: "580.173.02", cuda: "13.0",
			uuid: "GPU-0cddbf68-70f0-0aa4-7f92-a624e48fef64", computeMode: "Default",
			persistenceMode: "Enabled", pstate: "P0", pciBusID: "0000000F:01:00.0",
			vbios: "9A.0B.1E.00.00", computeCap: "12.1", host: "dgx1",
		},
	}

	got := render(s)

	want := `# HELP dgx_unified_memory_total_bytes Total unified memory (host system RAM) available to the GPU.
# TYPE dgx_unified_memory_total_bytes gauge
dgx_unified_memory_total_bytes 104857600
# HELP dgx_unified_memory_used_bytes Unified memory in active use (total - available).
# TYPE dgx_unified_memory_used_bytes gauge
dgx_unified_memory_used_bytes 52428800
# HELP dgx_unified_memory_available_bytes Unified memory available for new allocations.
# TYPE dgx_unified_memory_available_bytes gauge
dgx_unified_memory_available_bytes 52428800
# HELP dgx_unified_memory_gpu_used_bytes Unified memory held by active GPU processes (sum of compute+graphics).
# TYPE dgx_unified_memory_gpu_used_bytes gauge
dgx_unified_memory_gpu_used_bytes 116678197248
# HELP dgx_unified_memory_process_used_bytes Unified memory held by one GPU process (compute C or graphics G context).
# TYPE dgx_unified_memory_process_used_bytes gauge
dgx_unified_memory_process_used_bytes{pid="2408",type="G",process_name="/usr/lib/xorg/Xorg"} 18874368
dgx_unified_memory_process_used_bytes{pid="469190",type="C",process_name="VLLM::EngineCore"} 116659322880
# HELP dgx_gpu_utilization_ratio GPU compute utilization (nvidia-smi GPU-Util) as a ratio.
# TYPE dgx_gpu_utilization_ratio gauge
dgx_gpu_utilization_ratio 0.000000
# HELP dgx_gpu_memory_controller_ratio Memory controller utilization ratio.
# TYPE dgx_gpu_memory_controller_ratio gauge
dgx_gpu_memory_controller_ratio 0.000000
# HELP dgx_gpu_temperature_celsius GPU temperature in Celsius. nan only when nvidia-smi could not answer.
# TYPE dgx_gpu_temperature_celsius gauge
dgx_gpu_temperature_celsius 52.0
# HELP dgx_gpu_power_draw_watts GPU power draw in watts. nan only when nvidia-smi could not answer.
# TYPE dgx_gpu_power_draw_watts gauge
dgx_gpu_power_draw_watts 9.43
# HELP dgx_gpu_compute_apps Number of processes with a compute context on the GPU.
# TYPE dgx_gpu_compute_apps gauge
dgx_gpu_compute_apps 1
# HELP dgx_gpu_info Static GPU info.
# TYPE dgx_gpu_info gauge
dgx_gpu_info{name="NVIDIA GB10",driver="580.173.02",cuda="13.0",uuid="GPU-0cddbf68-70f0-0aa4-7f92-a624e48fef64",pci_bus_id="0000000F:01:00.0",host="dgx1",vbios="9A.0B.1E.00.00",compute_cap="12.1",pstate="P0",compute_mode="Default"} 1
# HELP dgx_gpu_pstate GPU performance state (NVIDIA P-state).
# TYPE dgx_gpu_pstate gauge
dgx_gpu_pstate{pstate="P0"} 1
# HELP dgx_gpu_compute_mode GPU compute mode.
# TYPE dgx_gpu_compute_mode gauge
dgx_gpu_compute_mode{mode="Default"} 1
# HELP dgx_gpu_compute_mode_enabled 1 if compute mode is 'Exclusive_Process'.
# TYPE dgx_gpu_compute_mode_enabled gauge
dgx_gpu_compute_mode_enabled 0
# HELP dgx_gpu_persistence_mode GPU persistence mode.
# TYPE dgx_gpu_persistence_mode gauge
dgx_gpu_persistence_mode{mode="Enabled"} 1
# HELP dgx_gpu_persistence_mode_enabled 1 if persistence mode is Enabled (0 if Disabled).
# TYPE dgx_gpu_persistence_mode_enabled gauge
dgx_gpu_persistence_mode_enabled 1
# HELP dgx_collect_success Whether collection succeeded (1) or not (0).
# TYPE dgx_collect_success gauge
dgx_collect_success 1
`
	if got != want {
		t.Errorf("render mismatch:\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

// TestRenderAlwaysEmitsCrucialMetrics guards the guarantee that temperature,
// power and the memory controller ratio are present as series even when every
// source came back unknown. An absent series reads as "no data", which is
// indistinguishable from an exporter that is not deployed at all.
func TestRenderAlwaysEmitsCrucialMetrics(t *testing.T) {
	out := render(snapshot{total: 1024, avail: 512, used: 512})

	for _, want := range []string{
		"dgx_gpu_temperature_celsius nan\n",
		"dgx_gpu_power_draw_watts nan\n",
		"dgx_gpu_memory_controller_ratio nan\n",
		"dgx_gpu_utilization_ratio nan\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderDegradedMarksFailureUnreachable guards dgx_collect_success 0: it is
// the signal operators alert on, so it must actually fire on a bad cycle.
func TestRenderDegradedMarksFailure(t *testing.T) {
	out := render(snapshot{errs: []string{"GPU temperature unavailable"}})

	if !strings.Contains(out, "dgx_collect_success 0\n") {
		t.Errorf("expected dgx_collect_success 0, got:\n%s", out)
	}
	if !strings.Contains(out, "# collector error: GPU temperature unavailable\n") {
		t.Errorf("expected the failure reason as a comment, got:\n%s", out)
	}
	// A degraded cycle must still serve the metrics that did answer.
	if !strings.Contains(out, "dgx_gpu_temperature_celsius") {
		t.Error("degraded cycle must still render the temperature series")
	}
}

func TestRenderMemoryUnknownIsNanNotZero(t *testing.T) {
	out := render(snapshot{memOK: false, total: 0, avail: 0, used: 0})
	for _, want := range []string{
		"dgx_unified_memory_total_bytes nan\n",
		"dgx_unified_memory_used_bytes nan\n",
		"dgx_unified_memory_available_bytes nan\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("unreadable meminfo must render %q, not a fabricated 0; got:\n%s", want, out)
		}
	}
}

func TestCollectHealthy(t *testing.T) {
	mem := writeMeminfo(t, meminfoFixture)
	c := newCollectorWithSmi(mem, fakeSmi{table: fakeTable, tele: fakeTele, info: fakeInfo, apps: fakeApps}.run)

	out, err := c.collect()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{
		"dgx_gpu_temperature_celsius 52.0\n",
		"dgx_gpu_power_draw_watts 9.43\n",
		"dgx_gpu_memory_controller_ratio 0.000000\n",
		"dgx_collect_success 1\n",
		`cuda="13.0"`,
		`uuid="GPU-0cddbf68-70f0-0aa4-7f92-a624e48fef64"`,
		"dgx_gpu_compute_apps 1\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "# collector error:") {
		t.Errorf("healthy cycle must not report errors:\n%s", out)
	}
	if len(c.lastErrs) != 0 {
		t.Errorf("lastErrs = %v, want empty", c.lastErrs)
	}
}

// TestCollectSmiUnavailableIsDegradedNotSilent is the regression test for the
// false-success bug: with nvidia-smi gone but meminfo readable, the exporter
// must serve what it has and flag dgx_collect_success 0.
func TestCollectSmiUnavailableIsDegradedNotSilent(t *testing.T) {
	mem := writeMeminfo(t, meminfoFixture)
	c := newCollectorWithSmi(mem, fakeSmi{err: errSmiUnavailable}.run)

	out, err := c.collect()
	if err != nil {
		t.Fatalf("degraded cycle must still serve data, got error: %v", err)
	}
	if !strings.Contains(out, "dgx_collect_success 0\n") {
		t.Errorf("expected dgx_collect_success 0 when nvidia-smi is gone, got:\n%s", out)
	}
	if !strings.Contains(out, "dgx_gpu_temperature_celsius nan\n") {
		t.Errorf("temperature series must still be present as nan, got:\n%s", out)
	}
	if !strings.Contains(out, "dgx_unified_memory_total_bytes 131072000000\n") {
		t.Errorf("meminfo-derived metrics must survive, got:\n%s", out)
	}
	if len(c.lastErrs) == 0 {
		t.Error("lastErrs must record the failure so main.go can log it")
	}
}

func TestCollectBothSourcesDeadIsFatal(t *testing.T) {
	c := newCollectorWithSmi("/nonexistent/meminfo", fakeSmi{err: errSmiUnavailable}.run)

	if out, err := c.collect(); err == nil {
		t.Fatalf("expected a fatal error when nothing is collectable, got:\n%s", out)
	}
	if len(c.lastErrs) == 0 {
		t.Error("lastErrs must be populated on the fatal path too")
	}
}

// TestCollectTempFallsBackToTable covers the second source for the crucial
// thermal metrics: the structured query answers nothing, the table row does.
func TestCollectTempFallsBackToTable(t *testing.T) {
	mem := writeMeminfo(t, meminfoFixture)
	c := newCollectorWithSmi(mem, fakeSmi{
		table: tableTempPower,
		tele:  "[N/A], [N/A], [N/A], [N/A]",
		info:  fakeInfo,
	}.run)

	out, err := c.collect()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{
		"dgx_gpu_temperature_celsius 48.0\n",
		"dgx_gpu_power_draw_watts 9.00\n",
		"dgx_gpu_utilization_ratio 0.000000\n",
		"dgx_collect_success 1\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}

// TestCollectMissingTempMarksFailure pins the promise that losing temperature is
// never reported as success.
func TestCollectMissingTempMarksFailure(t *testing.T) {
	mem := writeMeminfo(t, meminfoFixture)
	c := newCollectorWithSmi(mem, fakeSmi{
		table: "",
		tele:  "N/A, N/A, 0, 0",
		info:  fakeInfo,
	}.run)

	out, err := c.collect()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "dgx_collect_success 0\n") {
		t.Errorf("losing temperature must clear dgx_collect_success, got:\n%s", out)
	}
	if !strings.Contains(out, "# collector error: GPU temperature unavailable\n") {
		t.Errorf("expected the temperature loss to be named, got:\n%s", out)
	}
}

// TestCollectOptionalFieldNAStaysHealthy pins the documented asymmetry: an
// unsupported optional field renders nan without failing the cycle, because
// failing on a field the hardware legitimately lacks would leave
// dgx_collect_success permanently 0 and therefore useless.
func TestCollectOptionalFieldNAStaysHealthy(t *testing.T) {
	mem := writeMeminfo(t, meminfoFixture)
	c := newCollectorWithSmi(mem, fakeSmi{
		table: fakeTable,
		tele:  "52, 9.43, 0, N/A",
		info:  fakeInfo,
	}.run)

	out, err := c.collect()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "dgx_gpu_memory_controller_ratio nan\n") {
		t.Errorf("unsupported memory-controller field must render nan, got:\n%s", out)
	}
	if !strings.Contains(out, "dgx_collect_success 1\n") {
		t.Errorf("an unsupported optional field must not fail the cycle, got:\n%s", out)
	}
}

func TestCollectComputeAppsExcludedWhenSmiTableFails(t *testing.T) {
	mem := writeMeminfo(t, meminfoFixture)
	// Plain table exits non-zero-ish but is unavailable: compute apps must not
	// be counted from a source that never answered.
	c := newCollectorWithSmi(mem, fakeSmi{err: errSmiUnavailable}.run)
	out, _ := c.collect()
	if !strings.Contains(out, "dgx_gpu_compute_apps 0\n") {
		t.Errorf("compute apps must be 0 with no nvidia-smi, got:\n%s", out)
	}
}

func TestEscape(t *testing.T) {
	cases := []struct{ in, want string }{
		{`a\b`, `a\\b`},
		{`a"b`, `a\"b`},
		{`a\b"c`, `a\\b\"c`},
	}
	for _, c := range cases {
		if got := escape(c.in); got != c.want {
			t.Errorf("escape(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestFormatLabels(t *testing.T) {
	if got, want := formatLabels([]label{{"a", "1"}, {"b", "2"}}), `{a="1",b="2"}`; got != want {
		t.Errorf("formatLabels = %q, want %q", got, want)
	}
	if got := formatLabels(nil); got != "" {
		t.Errorf("formatLabels(nil) = %q, want empty", got)
	}
}

func TestErrorPayload(t *testing.T) {
	p := errorPayload(errForTest("boom"))
	want := "# HELP dgx_collect_success Whether collection succeeded (1) or not (0).\n" +
		"# TYPE dgx_collect_success gauge\n" +
		"dgx_collect_success 0\n" +
		"# collector error: boom\n"
	if p != want {
		t.Errorf("errorPayload mismatch:\n--- got ---\n%s\n--- want ---\n%s", p, want)
	}
}

func TestHTTPHandler(t *testing.T) {
	latest := "# HELP m m\n# TYPE m gauge\nm 1\n"
	mux := http.NewServeMux()
	// Mirror main.go: register the catch-all at "/" so that "/" and
	// "/metrics" both route to the same handler.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimRight(r.URL.Path, "/")
		if p == "" || p == "/metrics" {
			w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
			w.Header().Set("Content-Length", strconv.Itoa(len(latest)))
			w.WriteHeader(200)
			w.Write([]byte(latest))
			return
		}
		http.NotFound(w, r)
	})

	for _, tc := range []struct {
		path string
		code int
	}{
		{"/metrics", 200},
		{"/", 200},
		{"/metrics/", 200},
		{"/other", 404},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tc.path, nil))
		if rec.Code != tc.code {
			t.Errorf("%s status = %d, want %d", tc.path, rec.Code, tc.code)
		}
	}

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if ct := rec.Header().Get("Content-Type"); ct != "text/plain; version=0.0.4; charset=utf-8" {
		t.Errorf("content-type = %q", ct)
	}
	if body := rec.Body.String(); body != latest {
		t.Errorf("body = %q, want %q", body, latest)
	}
}

func TestProcessEquality(t *testing.T) {
	a := process{pid: "1", kind: "C", name: "x", bytes: 1}
	b := process{pid: "1", kind: "C", name: "x", bytes: 1}
	if !reflect.DeepEqual(a, b) {
		t.Errorf("expected equal processes")
	}
}

// helpers

type testErr string

func (e testErr) Error() string { return string(e) }

func errForTest(s string) error { return testErr(s) }
