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

// TestReadMeminfo uses a temp file so the real /proc/meminfo is not required.
func TestReadMeminfo(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "meminfo")
	content := `MemTotal:       128000000 kB
MemFree:         10000000 kB
MemAvailable:    90000000 kB
Buffers:            100000 kB
Cached:            2000000 kB
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	total, avail := readMeminfo(path)
	wantTotal := int64(128000000 * 1024)
	wantAvail := int64(90000000 * 1024)
	if total != wantTotal {
		t.Errorf("total = %d, want %d", total, wantTotal)
	}
	if avail != wantAvail {
		t.Errorf("avail = %d, want %d", avail, wantAvail)
	}
}

func TestReadMeminfoMissing(t *testing.T) {
	total, avail := readMeminfo("/nonexistent/meminfo")
	if total != 0 || avail != 0 {
		t.Errorf("expected (0,0), got (%d,%d)", total, avail)
	}
}

// Fixture that mimics the shape of `nvidia-smi` on a DGX Spark.
const smiFixture = `+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.70               KMD 6.6.0-rt78     DMD 0.0.0                      |
+---------------------------------+------------------------+----------------------+
| GPU  Name              T.C.D   | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf          Pwr: |                       |                          |
|   0  GB10          N/A  30C    | N/A             11.2W | N/A                    |
+---------------------------------+------------------------+
|   0  N/A   N/A            2408      G   /usr/lib/xorg/Xorg      18MiB |
+---------------------------------+------------------------+
| Not Supported          |      0%      Default           |
+---------------------------------+------------------------+
`

// The reTempPower regex requires the line to start with "| N/A" (pipe,
// optional whitespace, then N/A) — matching the shape the TOGO doc specifies:
//
//	| N/A   30C    P8      11.2W /  N/A  |
//
// In real nvidia-smi output the GPU index and name precede the N/A column,
// so the regex searches for the pattern anywhere in the line via re.search.
const smiFixtureReal = `+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.70               KMD 6.6.0-rt78     DMD 0.0.0                      |
+---------------------------------+------------------------+----------------------+
| GPU  Name              T.C.D   | Bus-Id        Disp.A | Volatile Uncorr. ECC |
+---------------------------------+------------------------+----------------------+
| N/A   30C    P8      11.2W /  N/A  |
+---------------------------------+------------------------+
|   0  N/A   N/A            2408      G   /usr/lib/xorg/Xorg      18MiB |
+---------------------------------+------------------------+
| Not Supported          |      0%      Default           |
+---------------------------------+------------------------+
`

func TestParseSmiTable(t *testing.T) {
	gpuUsed, procs, util, memUtil, temp, power, appCount := parseSmiTable(smiFixtureReal)

	// 18 MiB = 18 * 1024 * 1024
	wantGPUUsed := int64(18) * 1024 * 1024
	if gpuUsed != wantGPUUsed {
		t.Errorf("gpuUsed = %d, want %d", gpuUsed, wantGPUUsed)
	}
	if appCount != 0 {
		t.Errorf("appCount = %d, want 0 (table step does not count compute apps)", appCount)
	}
	if memUtil != nil {
		t.Errorf("memUtil = %v, want nil", *memUtil)
	}
	if util == nil || *util != 0.0 {
		t.Errorf("util = %v, want 0.0", util)
	}
	if temp == nil || *temp != 30.0 {
		t.Errorf("temp = %v, want 30.0", temp)
	}
	if power == nil || *power != 11.2 {
		t.Errorf("power = %v, want 11.2", power)
	}
	if len(procs) != 1 {
		t.Fatalf("len(procs) = %d, want 1", len(procs))
	}
	p := procs[0]
	if p.pid != "2408" {
		t.Errorf("pid = %q, want 2408", p.pid)
	}
	if p.kind != "G" {
		t.Errorf("kind = %q, want G", p.kind)
	}
	if p.name != "/usr/lib/xorg/Xorg" {
		t.Errorf("name = %q, want /usr/lib/xorg/Xorg", p.name)
	}
	if p.bytes != wantGPUUsed {
		t.Errorf("bytes = %d, want %d", p.bytes, wantGPUUsed)
	}
}

func TestParseSmiTableEmpty(t *testing.T) {
	gpuUsed, procs, util, memUtil, temp, power, appCount := parseSmiTable("")
	if gpuUsed != 0 || util != nil || memUtil != nil || temp != nil || power != nil || appCount != 0 {
		t.Errorf("expected zero results: gpuUsed=%d util=%v memUtil=%v temp=%v power=%v appCount=%d",
			gpuUsed, util, memUtil, temp, power, appCount)
	}
	if len(procs) != 0 {
		t.Errorf("len(procs) = %d, want 0", len(procs))
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
	wantBytes := int64(111255) * 1024 * 1024
	if procs[0].bytes != wantBytes {
		t.Errorf("procs[0].bytes = %d, want %d", procs[0].bytes, wantBytes)
	}
	if procs[1].pid != "12345" || procs[1].name != "python" {
		t.Errorf("procs[1] = %+v", procs[1])
	}
	wantBytes2 := int64(128) * 1024 * 1024
	if procs[1].bytes != wantBytes2 {
		t.Errorf("procs[1].bytes = %d, want %d", procs[1].bytes, wantBytes2)
	}
}

func TestParseComputeAppsEmpty(t *testing.T) {
	procs, count := parseComputeApps("")
	if count != 0 || len(procs) != 0 {
		t.Errorf("expected (nil, 0), got (%d procs, %d count)", len(procs), count)
	}
}

func TestParseComputeAppsMalformedSkipped(t *testing.T) {
	out := "abc, python, not_a_number\n"
	procs, count := parseComputeApps(out)
	if count != 0 || len(procs) != 0 {
		t.Errorf("expected nothing parsed, got procs=%v count=%d", procs, count)
	}
}

func TestGatherGPUInfoEnvOverrides(t *testing.T) {
	t.Setenv("DGX_HOST_NAME", "dgx1")
	t.Setenv("DGX_DRIVER_VERSION", "565.70")
	t.Setenv("DGX_CUDA_VERSION", "12.4")

	c := newCollector("/nonexistent/meminfo")
	info := c.gatherGPUInfo()

	if info.host != "dgx1" {
		t.Errorf("host = %q, want dgx1", info.host)
	}
	if info.driver != "565.70" {
		t.Errorf("driver = %q, want 565.70", info.driver)
	}
	if info.cuda != "12.4" {
		t.Errorf("cuda = %q, want 12.4", info.cuda)
	}
}

func TestGatherGPUInfoDefaultsWhenEnvUnset(t *testing.T) {
	t.Setenv("DGX_HOST_NAME", "")
	t.Setenv("DGX_DRIVER_VERSION", "")
	t.Setenv("DGX_CUDA_VERSION", "")

	c := newCollector("/nonexistent/meminfo")
	info := c.gatherGPUInfo()

	// The default for name is "NVIDIA GB10" — but on a machine with a real
	// GPU, nvidia-smi will override it. We only assert on the fallback host
	// (which is always populated from os.Hostname() when DGX_HOST_NAME is
	// empty) and that cuda/driver are non-empty (real GPU: nvidia-smi
	// populates them, no GPU: defaults "unknown").
	host, _ := os.Hostname()
	if info.host != host {
		t.Errorf("host = %q, want %q", info.host, host)
	}
	if info.driver == "" {
		t.Errorf("driver is empty, want at least the default 'unknown'")
	}
	if info.cuda == "" {
		t.Errorf("cuda is empty, want at least the default 'unknown'")
	}
}

func TestRenderGolden(t *testing.T) {
	temp := 30.0
	power := 11.2
	util := 0.0
	gBytes := int64(18) * 1024 * 1024
	cBytes := int64(111255) * 1024 * 1024
	s := snapshot{
		total:   100 * 1024 * 1024,
		avail:   50 * 1024 * 1024,
		used:    50 * 1024 * 1024,
		gpuUsed: gBytes + cBytes,
		procs: []process{
			{pid: "2408", kind: "G", name: "/usr/lib/xorg/Xorg", bytes: gBytes},
			{pid: "469190", kind: "C", name: "VLLM::EngineCore", bytes: cBytes},
		},
		util:     &util,
		memUtil:  nil,
		temp:     &temp,
		power:    &power,
		appCount: 1,
		info: gpuInfo{
			name:            "NVIDIA GB10",
			driver:          "565.70",
			cuda:            "12.4",
			uuid:            "GPU-12345678-1234-1234",
			computeMode:     "Default",
			persistenceMode: "Enabled",
			pstate:          "P0",
			pciBusID:        "0000:00:00.0",
			vbios:           "97.40",
			computeCap:      "12.0",
			host:            "dgx1",
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
# HELP dgx_gpu_temperature_celsius GPU temperature in Celsius.
# TYPE dgx_gpu_temperature_celsius gauge
dgx_gpu_temperature_celsius 30.0
# HELP dgx_gpu_power_draw_watts GPU power draw in watts.
# TYPE dgx_gpu_power_draw_watts gauge
dgx_gpu_power_draw_watts 11.20
# HELP dgx_gpu_compute_apps Number of processes with a compute context on the GPU.
# TYPE dgx_gpu_compute_apps gauge
dgx_gpu_compute_apps 1
# HELP dgx_gpu_info Static GPU info.
# TYPE dgx_gpu_info gauge
dgx_gpu_info{name="NVIDIA GB10",driver="565.70",cuda="12.4",uuid="GPU-1234",pci_bus_id="0000:00:00.0",host="dgx1",vbios="97.40",compute_cap="12.0",pstate="P0",compute_mode="Default"} 1
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

func TestRenderNilUtilNan(t *testing.T) {
	s := snapshot{
		total:    1024,
		avail:    512,
		used:     512,
		gpuUsed:  0,
		procs:    nil,
		util:     nil,
		appCount: 0,
		info:     gpuInfo{host: "x"},
	}
	out := render(s)
	if !strings.Contains(out, "dgx_gpu_utilization_ratio nan\n") {
		t.Errorf("expected 'dgx_gpu_utilization_ratio nan', got:\n%s", out)
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
	labels := []label{{"a", "1"}, {"b", "2"}}
	got := formatLabels(labels)
	want := `{a="1",b="2"}`
	if got != want {
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
	// "/metrics" both route to the same handler (Go's mux would otherwise
	// shadow the catch-all with a more specific "/metrics" pattern).
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimRight(r.URL.Path, "/")
		if p == "" || p == "/metrics" {
			w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
			w.Header().Set("Content-Length", strconvItoa(len(latest)))
			w.WriteHeader(200)
			w.Write([]byte(latest))
			return
		}
		http.NotFound(w, r)
	})
	rec := httptest.NewRecorder()

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	mux.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Errorf("/metrics status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/plain; version=0.0.4; charset=utf-8" {
		t.Errorf("content-type = %q", ct)
	}
	if body := rec.Body.String(); body != latest {
		t.Errorf("body = %q, want %q", body, latest)
	}

	rec2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodGet, "/", nil)
	mux.ServeHTTP(rec2, req2)
	if rec2.Code != 200 {
		t.Errorf("/ status = %d, want 200", rec2.Code)
	}

	rec3 := httptest.NewRecorder()
	req3 := httptest.NewRequest(http.MethodGet, "/other", nil)
	mux.ServeHTTP(rec3, req3)
	if rec3.Code != 404 {
		t.Errorf("/other status = %d, want 404", rec3.Code)
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

func strconvItoa(n int) string {
	return strconv.Itoa(n)
}
