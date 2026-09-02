# dgx-prometheus-exporter

A self-contained Prometheus exporter for NVIDIA DGX Spark (GB10) unified
memory and GPU telemetry. The GB10 has **no discrete VRAM** — `nvidia-smi`
reports memory as "Not Supported" — so all CUDA allocations share the host's
system RAM. This collector derives unified-memory metrics from
`/proc/meminfo` and GPU telemetry from `nvidia-smi`, then serves them on
`:9273/metrics`.

## Metrics

All 18 metrics below are emitted on **every** successful cycle. A metric whose
source could not answer is emitted as `nan` rather than omitted, so a lost
reading shows up as a gap in a graph instead of a series that silently
disappears from queries.

| Name | Type | Description | Labels |
|---|---|---|---|
| `dgx_unified_memory_total_bytes` | gauge | Total unified memory (host system RAM) available to the GPU. | — |
| `dgx_unified_memory_used_bytes` | gauge | Unified memory in active use (total - available). | — |
| `dgx_unified_memory_available_bytes` | gauge | Unified memory available for new allocations. | — |
| `dgx_unified_memory_gpu_used_bytes` | gauge | Unified memory held by active GPU processes (sum of compute+graphics). | — |
| `dgx_unified_memory_process_used_bytes` | gauge | Unified memory held by one GPU process (compute `C` or graphics `G` context). | `pid`, `type`, `process_name` |
| `dgx_gpu_utilization_ratio` | gauge | GPU compute utilization (`utilization.gpu`) as a ratio. | — |
| `dgx_gpu_memory_controller_ratio` | gauge | Memory controller utilization (`utilization.memory`) as a ratio. | — |
| `dgx_gpu_temperature_celsius` | gauge | GPU temperature in Celsius (`temperature.gpu`). | — |
| `dgx_gpu_power_draw_watts` | gauge | GPU power draw in watts (`power.draw`). | — |
| `dgx_gpu_compute_apps` | gauge | Number of processes with a compute context on the GPU. | — |
| `dgx_gpu_info` | gauge | Static GPU info. | `name`, `driver`, `cuda`, `uuid`, `pci_bus_id`, `host`, `vbios`, `compute_cap`, `pstate`, `compute_mode` |
| `dgx_spark_info` | gauge | DGX Spark host info. | `host` |
| `dgx_gpu_pstate` | gauge | GPU performance state (NVIDIA P-state). | `pstate` |
| `dgx_gpu_compute_mode` | gauge | GPU compute mode. | `mode` |
| `dgx_gpu_compute_mode_enabled` | gauge | 1 if compute mode is `Exclusive_Process`. | — |
| `dgx_gpu_persistence_mode` | gauge | GPU persistence mode. | `mode` |
| `dgx_gpu_persistence_mode_enabled` | gauge | 1 if persistence mode is `Enabled`. | — |
| `dgx_collect_success` | gauge | Whether the last collection cycle was fully healthy. | — |

Notes on value formats:

- Ratios (`*_ratio`) render as `%.6f`. Temperature renders as `%.1f`, power as
  `%.2f`. Byte counters render as integers.
- Any value whose source did not answer renders as `nan`. That includes the
  four `dgx_unified_memory_*` counters when `/proc/meminfo` was unreadable —
  they are never reported as a fabricated `0`.
- `uuid` is the full GPU UUID (`GPU-0cddbf68-...`), not a shortened form.
- `cuda` comes from the `nvidia-smi` banner; it is not a valid `--query-gpu`
  field. It reads `unknown` when the banner cannot be parsed, so set
  `DGX_CUDA_VERSION` if you need it pinned.
- The per-process metric's `type` label is `C` for compute (from
  `--query-compute-apps`) and `G` for graphics (from the human-readable table,
  which is the only source for graphics contexts).

## Collection and health reporting

Each cycle issues four `nvidia-smi` invocations: the human-readable table
(graphics contexts, banner CUDA version, and a fallback for the thermal
fields), a structured telemetry query
(`temperature.gpu,power.draw,utilization.gpu,utilization.memory`), a static
info query, and the compute-apps query. Telemetry comes from the structured
query in preference to the table, because the table truncates wide columns and
reflows with terminal width.

The `telemetry` and `info` queries are deliberately kept separate: merging them
would let one unrecognised info field fail the whole call and take the
temperature reading down with it.

`dgx_collect_success` has two failure modes, and both are reachable:

| State | What you get | `dgx_collect_success` |
|---|---|---|
| Healthy | all metrics with real values | `1` |
| Degraded — a source failed, but something answered | every metric still served; unknowns as `nan`; one `# collector error: <reason>` comment line per failed source | `0` |
| Dead — `/proc/meminfo` unreadable **and** `nvidia-smi` unexecutable | only the `dgx_collect_success 0` line plus the error | `0` |

Temperature and power are treated as load-bearing: if `nvidia-smi` runs but
neither the structured query nor the table yields a temperature, the cycle is
marked failed rather than passed off as a healthy idle GPU.

Optional fields are not load-bearing. A ratio that the hardware does not
implement — `utilization.memory` is "Not Supported" on many GPUs — renders as
`nan` while the cycle stays healthy, because failing the cycle for an
expected-absent field would make `dgx_collect_success` permanently `0` and
therefore useless. So `dgx_gpu_temperature_celsius nan` always implies
`dgx_collect_success 0`, whereas `dgx_gpu_memory_controller_ratio nan` does
not.

Degradation is logged to stderr on the transition only — once when a source
breaks, once when it recovers — so a persistently missing driver does not emit
a line every interval.

Recommended alerts:

```yaml
- alert: DGXExporterBlind
  expr: dgx_collect_success == 0
  for: 2m
- alert: DGXGPUTooHot
  expr: dgx_gpu_temperature_celsius > 85
  for: 5m
```

## CLI Flags & Environment Variables

| Flag | Shorthand | Environment Variable | Default | Meaning |
|---|---|---|---|---|
| `--port` | `-p` | `DGX_EXPORTER_PORT` | `9273` | HTTP listen port. |
| `--interval` | `-i` | `COLLECT_INTERVAL` | `10` | Seconds between collection cycles. Must be > 0. |
| `--addr` | `-a` | — | `0.0.0.0` | Listen address/host. |
| `--version` | `-v` | — | — | Print version and exit. |
| `--help` | `-h` | — | — | Print usage help and exit. |
| — | — | `DGX_HOST_NAME` | `os.Hostname()` | `host` label on `dgx_gpu_info` and `dgx_spark_info`. |
| — | — | `DGX_DRIVER_VERSION` | `""` | Authoritative driver version; overrides nvidia-smi. |
| — | — | `DGX_CUDA_VERSION` | `""` | Authoritative CUDA version; overrides the banner. |

An unparseable `DGX_EXPORTER_PORT` or `COLLECT_INTERVAL` is logged as a warning
and falls back to the default. A value that parses but is out of range — a
non-positive interval, or a port outside 1–65535 — is a fatal startup error
rather than a silent misconfiguration: a zero interval would spin, forking
`nvidia-smi` three times per pass with no pause.

Set `scrape_interval` at or above the collection interval. The exporter caches
the last cycle, so scraping faster than you collect yields repeated samples.

## Build

Requires Go ≥ 1.21 and Linux.

```sh
make build          # produces ./dgx-exporter
# or
go build -o dgx-exporter .
```

## Run

```sh
./dgx-exporter
# quick check
curl -s localhost:9273/metrics
```

Running directly on the DGX host is the simplest deployment and is the only way
to get the per-process metrics — see the container note below.

## Container

The image is a static binary on a `debian:bookworm-slim` base. The base is not
cosmetic: `nvidia-smi` is dynamically linked against the host's libc, so a
`scratch` image cannot exec it at all. The kernel reports `ENOENT` for the
missing ELF interpreter, the exporter treats that as "no data", and you get an
exporter that serves no GPU telemetry at all.

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to be installed on the host (`docker info --format '{{json .Runtimes}}'` should
list `nvidia`).

```sh
make docker         # tags dgx-exporter:<git-sha> and dgx-exporter:latest
docker run -d --name dgx-exporter \
  --gpus all \
  --pid=host \
  -p 9273:9273 \
  -e DGX_HOST_NAME=$(hostname) \
  dgx-exporter
```

- `--gpus all` is what injects `nvidia-smi`, the driver libraries and the
  `/dev/nvidia*` nodes. Without it the container has no GPU access and every
  GPU metric is `nan`.
- `--pid=host` is required for `dgx_unified_memory_process_used_bytes` and
  `dgx_gpu_compute_apps`. NVIDIA's userspace remaps PIDs into the container's
  namespace, so without it those queries legitimately return an empty list.
- Do **not** bind-mount `nvidia-smi` or `/usr/bin/nvidia-smi` from the host:
  `--gpus all` already provides it, and a host binary without its libraries
  will not run.
- Do **not** bind-mount `/proc/meminfo`. A container already reads the host's
  real `/proc/meminfo` (there is no cgroup-based masking unless you install
  lxcfs), so the mount adds nothing.
- `DGX_HOST_NAME` matters: the container's own hostname is a truncated
  container ID, which makes a poor `host` label.

`DGX_DRIVER_VERSION` and `DGX_CUDA_VERSION` are optional overrides for the
labels. Inside a container `nvidia-smi` reports what the injected compatibility
library advertises, which need not match the host driver — set these if you
alert on those label values.

### Compose

`docker-compose.yml` encodes the same recipe, including `pid: host` and the
NVIDIA device reservation:

```sh
DGX_HOST_NAME=$(hostname) docker compose up -d --build
curl -s localhost:9273/metrics
```

`DGX_HOST_NAME` is read from your shell (or a `.env` file, which is gitignored)
and falls back to `dgx-spark`. Everything else has a sane default; the two label
overrides are present but commented out.

## Prometheus scrape config

```yaml
scrape_configs:
  - job_name: dgx
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["<dgx-host>:9273"]
```

## Grafana dashboard

`grafana-dashboard.json` is a ready-to-import dashboard for the metrics above
(*NVIDIA DGX Spark (GB10) Telemetry*, uid `dgx-spark-gb10`). Import it via
**Dashboards → Import** and point the `Data Source` variable at your Prometheus.

It repeats one row per host (`DGX Spark — $host`), the `Host` dropdown being
fed by `label_values(dgx_spark_info, host)`. Only `dgx_spark_info` and
`dgx_gpu_info` carry a `host` label, so every other panel attaches it at query
time with `… * on(instance) group_left(host) dgx_spark_info`. Two consequences:

- hosts must have distinct `instance` labels (any normal scrape config gives
  you this), and
- `host` has to be meaningful — in a container that means setting
  `DGX_HOST_NAME`, otherwise the dropdown lists truncated container IDs.

Panels: exporter health, temperature, power draw, compute apps, device info,
unified-memory overview and history, GPU / memory-controller utilization,
per-process memory usage, and a hardware + driver table. The per-process panel
stays empty without `--pid=host` (see Container). The temperature panels turn
red at 85 °C, matching the `DGXGPUTooHot` alert above; the power panels top out
at 140 W because the GB10 reports no `power.limit` to read.

## Troubleshooting

- **`dgx_collect_success 0`** — a source failed. Read the
  `# collector error: ...` comment lines at the bottom of `/metrics`; they name
  what was lost. Then run `nvidia-smi` manually to check that the driver is
  loaded and the GPU is reachable.
- **`dgx_gpu_temperature_celsius nan` or `dgx_gpu_power_draw_watts nan`** — the
  exporter could not read the thermal fields, and `dgx_collect_success` is `0`
  to match. If you ever see either of them `nan` while `dgx_collect_success` is
  `1`, that is a bug worth reporting.
- **`nan` on GPU metrics inside a container** — almost always a missing
  `--gpus all`, or a runtime image with no libc (see Container above). Check
  with `docker exec <ctr> nvidia-smi`.
- **`dgx_gpu_compute_apps 0` while a job is clearly running** — missing
  `--pid=host`.
- **Single GPU** — the exporter reads GPU 0 only. Multi-GPU hosts will only see
  the first GPU.

## License

See the [LICENSE](LICENSE) file.
