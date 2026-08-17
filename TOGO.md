# TOGO — Port `dgx_exporter.py` to Go

This document is the working plan for an AI agent to convert the existing
`dgx_exporter.py` (NVIDIA DGX Spark / GB10 Prometheus exporter) into a
self-contained Go program, and to produce the accompanying documentation and
build tooling. The conversion must be a **faithful port**: same metrics, same
HELP text, same label sets, same formatting, same environment variables, same
error behavior.

Source of truth: `dgx_exporter.py` in this repository. If any statement below
contradicts the Python file, the Python file wins and this plan should be
updated.

---

## 1. Goal

Build, in this repository, a Go implementation that exposes the identical
Prometheus metrics on `:9273/metrics` (configurable), collected from
`/proc/meminfo` and `nvidia-smi` on a DGX Spark (GB10) host, plus:

- a `Makefile` (build / test / lint / docker / clean targets),
- a `README.md` documenting the exporter (metrics, env vars, usage),
- unit tests covering all parsing and the HTTP handler,
- an optional `Dockerfile` for containerized deployment.

Constraints:

- **Linux only** — the program reads `/proc/meminfo` and shells out to
  `nvidia-smi`; it does not need to compile or run on other platforms.
- **Single GPU assumption** — the Python code only ever parses GPU 0
  (`nvidia-smi` rows starting `| 0 N/A N/A ...`). Keep this.
- **Zero third-party dependencies** — the port uses only the Go standard
  library (`net/http`, `os/exec`, `regexp`, `encoding/csv`, `bufio`, `sync`,
  `time`, `strconv`, `fmt`, `os`, `log`/`slog`). Rationale: the Python
  exporter renders the Prometheus text format by hand with no libraries, and
  the exact same approach in Go gives byte-for-byte output parity with zero
  deps. Do **not** add `prometheus/client_golang`; its float formatting
  (`strconv 'g'`, shortest form) differs from Python's fixed-precision output
  (e.g. `0.500000` vs `0.5`) and would break parity. `go.mod` must end up with
  no `require` lines.
- **Byte-level output parity** where the Python output is deterministic.
  The one unavoidable difference is the per-process series, which reflect
  whatever processes are running at collect time — that is expected.

---

## 2. Target repository layout

```
dgx_exporter.py        # existing, keep as-is (reference)
README.md              # NEW: rewrite from the current 1-line placeholder
TOGO.md                # this plan
Makefile               # NEW
Dockerfile             # NEW (optional but recommended)
go.mod                 # NEW
go.sum                 # NEW (empty of requires; may not be created — fine)
main.go                # NEW: entry point, env parsing, HTTP server, loop
collector.go           # NEW: all collection/parsing/render logic
collector_test.go      # NEW: unit tests with fixture strings
```

Flat package layout is intentional: the whole exporter is ~400 lines of Go,
small enough for a single package. Keep `dgx_exporter.py` in the repo as the
reference implementation (do not delete it).

Add to the existing `.gitignore` (it is already Go-oriented):
```
# built binary
/dgx-exporter
```

---

## 3. Module setup

`go.mod`:

```
module dgx-prometheus-exporter

go 1.21
```

(`go 1.21` is a conservative floor; the installed toolchain is newer. No
`require` lines. If `go mod tidy` still produces a `go.sum`, it will be empty —
do not force one.)

---

## 4. Behavior contract (what the port must reproduce)

### 4.1 Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DGX_EXPORTER_PORT` | `9273` | HTTP listen port |
| `COLLECT_INTERVAL` | `10` | Seconds between collection cycles |
| `DGX_HOST_NAME` | `os.Hostname()` | `host` label on `dgx_gpu_info` |
| `DGX_DRIVER_VERSION` | `""` | Authoritative driver version; overrides nvidia-smi |
| `DGX_CUDA_VERSION` | `""` | Authoritative CUDA version (nvidia-smi reports N/A in container) |

Ports are `strconv.Atoi` of the env values; on parse error fall back to the
default (Python's `int()` would raise, but a robust port should not crash —
log a warning and use the default).

### 4.2 Collection sources

1. **`/proc/meminfo`** — `MemTotal` and `MemAvailable`, values in kB,
   multiplied by 1024 to bytes. Missing/unreadable file ⇒ both `0` (the
   Python code swallows `OSError` and returns `(0, 0)`).
2. **`nvidia-smi`** (bare) — human-readable table, parsed for:
   - temperature + power via regex `\|\s*N/A\s+(\d+)C\s+\S+\s+([\d.]+)W\s*/\s*\S+\s*\|`,
   - GPU utilization via regex `Not Supported\s*\|\s+(\d+)%` → util / 100,
   - graphics (type `G`) context rows via regex
     `^\|\s+0\s+N/A\s+N/A\s+(\d+)\s+G\s+(.*?)\s+(\d+)(MiB|GiB)\s*\|$` →
     `(pid, name, bytes)`, where `GiB` multiplies by 1024 and bytes are
     `amount * 1024 * 1024`.
3. **`nvidia-smi --query-compute-apps=pid,process_name,used_memory
   --format=csv,noheader,nounits`** — compute (`C`) contexts. Each CSV row is
   `pid, name, used_memory_MiB`; MiB × 1024 × 1024 → bytes. Counts toward
   `dgx_gpu_compute_apps`. Rows that fail to parse are skipped (Python
   catches `ValueError`). Blank lines are ignored. Compute contexts are parsed
   via CSV, not the table, because nvidia-smi truncates large values in the
   table (e.g. `11125...` for 111255 MiB).
4. **`nvidia-smi --query-gpu=name,driver_version,uuid,compute_mode,
   persistence_mode,pstate,pci.bus_id,vbios_version,compute_cap
   --format=csv,noheader,nounits`** — one line of 9 comma-separated values;
   strip whitespace around each. If fewer than 9 parts, keep defaults.

Semantics for nvidia-smi invocations (mirror Python's `check=False` +
captured stdout):

- If the `nvidia-smi` binary does not exist (Go: `exec.LookPath` fails / the
  `exec.Error` from `Output()`), return the zero-value results — do not fail
  the whole collection. This is the port of Python's `except OSError`.
- If `nvidia-smi` runs but exits non-zero, still parse whatever stdout was
  produced (Python parses `out` regardless of the return code).

### 4.3 Static GPU info

Defaults (used until/unless nvidia-smi fills them):

```
name="NVIDIA GB10"  driver="unknown"  cuda="unknown"  uuid="unknown"
compute_mode="unknown"  persistence_mode="unknown"  pstate="unknown"
pci_bus_id="unknown"  vbios="unknown"  compute_cap="unknown"  host="unknown"
```

Then, in order:

1. nvidia-smi values overwrite the first 9 fields (if ≥9 CSV parts).
2. `host` = `DGX_HOST_NAME` (trimmed) if non-empty, else `os.Hostname()`.
3. `driver` = `DGX_DRIVER_VERSION` (trimmed) if non-empty.
4. `cuda` = `DGX_CUDA_VERSION` (trimmed) if non-empty.

### 4.4 Metrics — exact HELP text, labels, and value formats

All metrics are gauges. Ratio/`None`-able values follow Python's rules:

- `util` present → `%.6f`; absent → literal `nan`.
- `mem_util` is never actually reported on GB10 ("Not Supported"), but the
  gauge is only emitted when non-nil.
- `temp` → `%.1f`; `power` → `%.2f`; only emitted when present.
- Integer-valued gauges render as plain integers.

| Metric | HELP text (exact) | Labels / value |
|---|---|---|
| `dgx_unified_memory_total_bytes` | `Total unified memory (host system RAM) available to the GPU.` | total |
| `dgx_unified_memory_used_bytes` | `Unified memory in active use (total - available).` | total - avail |
| `dgx_unified_memory_available_bytes` | `Unified memory available for new allocations.` | avail |
| `dgx_unified_memory_gpu_used_bytes` | `Unified memory held by active GPU processes (sum of compute+graphics).` | gpu_used |
| `dgx_unified_memory_process_used_bytes` | `Unified memory held by one GPU process (compute C or graphics G context).` | one HELP/TYPE block, then one line per process: `{pid="…",type="G"\|"C",process_name="…"}` |
| `dgx_gpu_utilization_ratio` | `GPU compute utilization (nvidia-smi GPU-Util) as a ratio.` | `%.6f` or `nan` |
| `dgx_gpu_memory_controller_ratio` | `Memory controller utilization ratio.` | `%.6f` or `nan`; omit if nil |
| `dgx_gpu_temperature_celsius` | `GPU temperature in Celsius.` | `%.1f`; omit if nil |
| `dgx_gpu_power_draw_watts` | `GPU power draw in watts.` | `%.2f`; omit if nil |
| `dgx_gpu_compute_apps` | `Number of processes with a compute context on the GPU.` | count |
| `dgx_gpu_info` | `Static GPU info.` | `{name,driver,cuda,uuid,pci_bus_id,host,vbios,compute_cap,pstate,compute_mode}` all = 1 |
| `dgx_gpu_pstate` | `GPU performance state (NVIDIA P-state).` | `{pstate}` = 1 |
| `dgx_gpu_compute_mode` | `GPU compute mode.` | `{mode}` = 1 |
| `dgx_gpu_compute_mode_enabled` | `1 if compute mode is 'Exclusive_Process'.` | 1 if `compute_mode == "Exclusive_Process"` else 0 |
| `dgx_gpu_persistence_mode` | `GPU persistence mode.` | `{mode}` = 1 |
| `dgx_gpu_persistence_mode_enabled` | `1 if persistence mode is Enabled (0 if Disabled).` | 1 if `persistence_mode` lowercased starts with `enabled` else 0 |
| `dgx_collect_success` | `Whether collection succeeded (1) or not (0).` | 1 |

Label-order caveat (important for parity): the Python code emits labels in
dict insertion order — for `dgx_gpu_info` exactly the order in the table
above. Go maps iterate in random order, so **do not** build label strings from
a `map[string]string`. Use an ordered slice of `(name, value)` pairs or hard
coded per-metric label templates. Label escaping: replace `\` with `\\` then
`"` with `\"` (Python does backslash first, then quote).

`uuid` is truncated to its first 8 characters in `dgx_gpu_info`.

### 4.5 Exact output shape

- Each gauge contributes `# HELP`, `# TYPE`, then the value line(s).
- The per-process metric emits one `# HELP` + `# TYPE` block followed by N
  labeled series lines.
- The whole payload ends with a single trailing newline.
- On any error during collection, the served payload is **exactly**:
  ```
  # HELP dgx_collect_success Whether collection succeeded (1) or not (0).
  # TYPE dgx_collect_success gauge
  dgx_collect_success 0
  # collector error: <error message>
  ```
  (note: no trailing blank line; the error line ends with a newline).
- Before the first successful collection, the served payload is the literal
  string `# dgx custom collector starting...\n`.

### 4.6 Concurrency and update model

Same architecture as the Python original:

- A background goroutine runs the collection loop: collect → replace the
  cached payload → `sleep(INTERVAL)`.
- An HTTP handler serves the cached payload instantly (no nvidia-smi run per
  scrape), so the served text is at most `COLLECT_INTERVAL` seconds stale —
  exactly like Python.
- The payload is replaced atomically. Use `sync.RWMutex` (or
  `atomic.Pointer[string]`) so the handler never reads a half-written payload.

### 4.7 HTTP server

- Listen on `0.0.0.0:<PORT>` (bind all interfaces).
- `GET /`, `GET /metrics`, and any path that strips to those (`self.path.rstrip("/")`
  in `("", "/metrics")` or `self.path == "/metrics"`) → `200`.
- Response headers exactly:
  - `Content-Type: text/plain; version=0.0.4; charset=utf-8`
  - `Content-Length: <len>`
- Any other path → `404`.
- No per-request logging (Python silences it). `http.Server` errors may be
  logged to stderr; that is fine.

---

## 5. Implementation spec per file

### 5.1 `collector.go`

Type outline (single package, unexported helpers):

```go
type process struct {
    pid  string
    kind string // "G" graphics or "C" compute
    name string
    bytes int64
}

type gpuInfo struct {
    name, driver, cuda, uuid, computeMode, persistenceMode string
    pstate, pciBusID, vbios, computeCap, host              string
}

type snapshot struct {
    total, avail, used, gpuUsed int64
    procs   []process
    util    *float64 // nil = N/A
    memUtil *float64
    temp    *float64
    power   *float64
    appCount int
    info    gpuInfo
}
```

Functions (name them as you like; behavior is what matters):

- `readMeminfo() (total, avail int64)` — parse `/proc/meminfo`; kB × 1024.
  Use `bufio.Scanner`, split each line on `:`, extract the first digit run
  with a compiled `regexp` (`[0-9]+`) — do not assume the units field is
  exactly `kB`.
- `runNvidiaSmi(args ...string) (string, error)` — `exec.Command("nvidia-smi",
  args...).Output()`. Map "binary not found" to a distinct error (or return
  `("", err)` and let callers treat any error as "no data" except that
  non-zero exits still return stdout — implement by checking whether
  `stdout` is non-empty rather than by error type, which mirrors Python).
- `parseSmiTable(out string) (gpuUsed int64, procs []process, util, memUtil,
  temp, power *float64, appCount int)` — port of `_parse_nvidia_smi_table`.
  Three compiled regexes (4.2). Note: the graphics-row regex must be anchored
  `^...$` and match per line with `FindStringSubmatch`; the util regex matches
  a line containing `Not Supported | 0%`. The header temp/power regex uses the
  literal `N/A` temperature column. `appCount` is incremented in the
  compute-apps CSV step (not the table step).
- `parseComputeApps(out string) ([]process, int)` — `encoding/csv.NewReader`
  over the non-blank lines; per row `(pid, name, MiB)`; skip rows that fail
  `strconv.Atoi`/`ParseInt` on the third field.
- `gatherGPUInfo() gpuInfo` — port of `_nvidia_driver`, including env
  overrides (4.3).
- `collect() (string, error)` — orchestrates everything and renders the
  payload. Any error from a source (except the swallow-able nvidia-smi ones)
  is returned so the loop can emit the error payload.
- `render(s snapshot) string` — builds the exact text per section 4.4/4.5.
  Keep an ordered label helper:
  ```go
  type label struct{ k, v string }
  func gaugeLine(buf *strings.Builder, name string, labels []label, value string)
  ```
  and call it with labels in the fixed order from the table.

### 5.2 `main.go`

- Read env vars (4.1) with defaults and `log` (or `slog`) warnings on bad
  values.
- `var latest atomic.Pointer[string]` seeded with
  `"# dgx custom collector starting...\n"` (or a mutex-protected string).
- `go func()` loop: `text, err := collect(); if err != nil { latest.Store(errorPayload(err)) } else { latest.Store(text) } ; time.Sleep(interval)`.
- `http.HandleFunc` for `/` and `/metrics` (or a single handler checking the
  path per 4.7), writing headers/body from `latest.Load()`.
- `log.Fatal(http.ListenAndServe("0.0.0.0:"+port, mux))`.

### 5.3 `collector_test.go`

Fixture-driven unit tests (inline strings — do not depend on a live GPU):

1. `readMeminfo`: feed a fixture string via a temp file (or refactor
   `readMeminfo` to accept an `io.Reader`/path so tests can inject a fixture).
   Verify kB→bytes conversion.
2. `parseSmiTable`: fixture resembling real `nvidia-smi` output with:
   - a temp/power header row (`| N/A   30C    P8      4W /  N/A  |`),
   - a `Not Supported | 0%` util row,
   - one or more `G` rows (e.g. Xorg `18MiB`), and assert `gpuUsed`,
     `procs`, `util`, `temp`, `power`.
3. `parseComputeApps`: CSV fixture, assert byte conversion and appCount,
   and that a malformed row is skipped.
4. `gatherGPUInfo`: with env vars set (use `t.Setenv`), assert overrides.
5. `render`: golden-string test — call `render` with a hand-built `snapshot`
   and assert the full expected text, including the `dgx_gpu_info` label
   order, `nan` for a nil util, and `%`-precision values.
6. HTTP handler: `httptest.NewRecorder`; assert 200 + content-type + body on
   `/metrics` and `/`, and 404 on `/other`.

Do not write tests that exec real `nvidia-smi` (CI machines won't have it);
keep all parsing tests at the string level by structuring the functions to
take the command output as input.

### 5.4 `Makefile`

```make
BINARY   := dgx-exporter
VERSION  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS  := -s -w -X main.version=$(VERSION)

.PHONY: all build run test vet fmt lint docker clean

all: build

build:
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o $(BINARY) .

run:
	go run .

test:
	go test ./... -count=1

vet:
	go vet ./...

fmt:
	gofmt -l -w .

lint:
	@command -v golangci-lint >/dev/null 2>&1 \
		&& golangci-lint run ./... \
		|| echo "golangci-lint not installed; skipping (go vet covers static checks)"

docker:
	docker build -t dgx-exporter:$(VERSION) .

clean:
	rm -f $(BINARY)
```

Notes for the agent:
- `main.go` must declare `var version = "dev"` (the `-X` target) — set it to
  `VERSION` at build time; if you prefer, also surface it in a log line at
  startup.
- `gofmt -l -w .` must produce no output after formatting (code must be
  gofmt-clean).

### 5.5 `README.md`

Rewrite the current placeholder (`# dgx-prometheus-exporter`) into a full
document. Required sections, in order:

1. **Title + one-paragraph description** — exporter for NVIDIA DGX Spark
   (GB10) unified memory + GPU telemetry; note the GB10 has no discrete VRAM
   and nvidia-smi reports memory as "Not Supported", hence the unified-memory
   metrics derived from `/proc/meminfo`.
2. **Metrics** — table of every metric (name, type, description, labels)
   mirroring 4.4. Mention the `nan` behavior for N/A values and the
   per-process metric's `type` label (`C` = compute, `G` = graphics).
3. **Environment variables** — table from 4.1.
4. **Build** — `make build`, `go build`, output binary name `dgx-exporter`;
   Go ≥ 1.21, Linux required.
5. **Run** — `make run` / `./dgx-exporter`; quick check
   `curl -s localhost:9273/metrics`.
6. **Container** — `make docker`, `docker run -p 9273:9273 ...` with the
   `DGX_*` env vars; note `--privileged`-free: it only needs the host's
   `nvidia-smi` (mount/bind it if not present inside the image) and
   `/proc/meminfo`.
7. **Prometheus scrape config** — minimal `scrape_configs` snippet with
   `job_name: dgx`, `scrape_interval: 15s`, `metrics_path: /metrics`.
8. **Troubleshooting** — expected N/A values (`nan` output), why
   `dgx_collect_success` may be `0` (run `nvidia-smi` manually), single-GPU
   assumption.
9. **License** — reference the existing `LICENSE` file.

### 5.6 `Dockerfile` (recommended)

Multi-stage, static binary, non-root:

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.21-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /dgx-exporter .

FROM scratch
COPY --from=build /dgx-exporter /dgx-exporter
EXPOSE 9273
ENV DGX_EXPORTER_PORT=9273 COLLECT_INTERVAL=10
ENTRYPOINT ["/dgx-exporter"]
```

Note for the agent: `nvidia-smi` will not exist inside a `scratch` image; in
the README, instruct users to bind-mount it, e.g.
`-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro` — but a `scratch` image has no
shell, so `/usr/bin/nvidia-smi` must be a static binary. If that is a
concern, prefer `FROM debian:bookworm-slim` with
`apt-get install -y nvidia-driver-bin`… **do not over-engineer**: choose
`scratch` + mount and document the requirement, or `debian:bookworm-slim`
and document installing nvidia-smi. State the choice in the README.

---

## 6. Order of work (agent execution sequence)

1. `go mod init dgx-prometheus-exporter` (or write `go.mod` by hand).
2. Write `collector.go` (parsing → snapshot → render), then `main.go`.
3. Write `collector_test.go`; run `go test ./...` until green.
4. Write `Makefile`, `Dockerfile`, rewrite `README.md`, extend `.gitignore`.
5. Full verification pass (below).

## 7. Verification checklist (must all pass)

- `gofmt -l .` → empty output.
- `go vet ./...` → clean.
- `go test ./... -count=1` → all pass.
- `make build` → produces `./dgx-exporter`.
- `go mod tidy` → no `require` lines added.
- Manual smoke test **on a DGX host** (if available): run the binary,
  `curl -s localhost:9273/metrics`, and confirm:
  - metric names/HELP strings match the Python output (diff the HELP/TYPE
    lines),
  - `dgx_gpu_info` label order matches 4.4,
  - ratios render `%.6f`, temp `%.1f`, power `%.2f`,
  - `dgx_unified_memory_process_used_bytes` has both `C` and `G` rows as
    applicable.
- Manual error test: run with `PATH` that hides `nvidia-smi` (e.g.
  `PATH=/nonexistent ./dgx-exporter`) → `/metrics` returns the exact error
  payload from 4.5.
- If a real DGX is unavailable, state that clearly in the final report; the
  fixture tests are the acceptance gate.

## 8. Out of scope / explicit non-goals

- No multi-GPU support (Python never parses GPUs other than 0).
- No TLS, no auth, no config file — env vars only, like the original.
- No `client_golang` / no external dependencies.
- No changes to the existing `dgx_exporter.py`; it stays as the reference.
- No Windows/macOS support.
