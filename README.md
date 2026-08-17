# dgx-prometheus-exporter

A self-contained Prometheus exporter for NVIDIA DGX Spark (GB10) unified
memory and GPU telemetry. The GB10 has **no discrete VRAM** — `nvidia-smi`
reports memory as "Not Supported" — so all CUDA allocations share the host's
system RAM. This collector derives unified-memory metrics from
`/proc/meminfo` and GPU telemetry from `nvidia-smi`, then serves them on
`:9273/metrics`.

## Metrics

| Name | Type | Description | Labels |
|---|---|---|---|
| `dgx_unified_memory_total_bytes` | gauge | Total unified memory (host system RAM) available to the GPU. | — |
| `dgx_unified_memory_used_bytes` | gauge | Unified memory in active use (total - available). | — |
| `dgx_unified_memory_available_bytes` | gauge | Unified memory available for new allocations. | — |
| `dgx_unified_memory_gpu_used_bytes` | gauge | Unified memory held by active GPU processes (sum of compute+graphics). | — |
| `dgx_unified_memory_process_used_bytes` | gauge | Unified memory held by one GPU process (compute `C` or graphics `G` context). | `pid`, `type`, `process_name` |
| `dgx_gpu_utilization_ratio` | gauge | GPU compute utilization (nvidia-smi GPU-Util) as a ratio. | — |
| `dgx_gpu_memory_controller_ratio` | gauge | Memory controller utilization ratio. | — |
| `dgx_gpu_temperature_celsius` | gauge | GPU temperature in Celsius. | — |
| `dgx_gpu_power_draw_watts` | gauge | GPU power draw in watts. | — |
| `dgx_gpu_compute_apps` | gauge | Number of processes with a compute context on the GPU. | — |
| `dgx_gpu_info` | gauge | Static GPU info. | `name`, `driver`, `cuda`, `uuid`, `pci_bus_id`, `host`, `vbios`, `compute_cap`, `pstate`, `compute_mode` |
| `dgx_gpu_pstate` | gauge | GPU performance state (NVIDIA P-state). | `pstate` |
| `dgx_gpu_compute_mode` | gauge | GPU compute mode. | `mode` |
| `dgx_gpu_compute_mode_enabled` | gauge | 1 if compute mode is `Exclusive_Process`. | — |
| `dgx_gpu_persistence_mode` | gauge | GPU persistence mode. | `mode` |
| `dgx_gpu_persistence_mode_enabled` | gauge | 1 if persistence mode is `Enabled`. | — |
| `dgx_collect_success` | gauge | Whether the last collection succeeded (1) or not (0). | — |

Notes on value formats:
- Ratios (`*_ratio`) render as `%.6f`. When N/A they render as `nan`.
- Temperature renders as `%.1f`, power as `%.2f`.
- The per-process metric's `type` label is `C` for compute (from
  `--query-compute-apps`) and `G` for graphics (from the human-readable
  table).

## CLI Flags & Environment Variables

| Flag | Shorthand | Environment Variable | Default | Meaning |
|---|---|---|---|---|
| `--port` | `-p` | `DGX_EXPORTER_PORT` | `9273` | HTTP listen port. |
| `--interval` | `-i` | `COLLECT_INTERVAL` | `10` | Seconds between collection cycles. |
| `--addr` | `-a` | — | `0.0.0.0` | Listen address/host. |
| `--version` | `-v` | — | — | Print version and exit. |
| `--help` | `-h` | — | — | Print usage help and exit. |
| — | — | `DGX_HOST_NAME` | `os.Hostname()` | `host` label on `dgx_gpu_info`. |
| — | — | `DGX_DRIVER_VERSION` | `""` | Authoritative driver version; overrides nvidia-smi. |
| — | — | `DGX_CUDA_VERSION` | `""` | Authoritative CUDA version (nvidia-smi reports N/A in container). |

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

## Container

```sh
make docker
docker run -d -p 9273:9273 \
  -e DGX_HOST_NAME=$(hostname) \
  -e DGX_DRIVER_VERSION=565.70 \
  -e DGX_CUDA_VERSION=12.4 \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /proc/meminfo:/proc/meminfo:ro \
  dgx-exporter
```

The image is a static binary in a `scratch` layer. Because `scratch` has no
shell, you must bind-mount `nvidia-smi` (and its shared library dependencies
if not statically linked) and `/proc/meminfo` into the container. Alternatively,
use a `debian:bookworm-slim` base and install `nvidia-driver-bin` inside the
image.

## Prometheus scrape config

```yaml
scrape_configs:
  - job_name: dgx
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["<dgx-host>:9273"]
```

## Troubleshooting

- **`nan` output** — When `nvidia-smi` reports a field as N/A (memory
  controller utilization on GB10, for example), the Go port emits `nan`,
  matching Python's `float("nan")` rendering.
- **`dgx_collect_success 0`** — The exporter could not collect. Run
  `nvidia-smi` manually on the host to check whether the driver is loaded
  and the GPU is reachable.
- **Single GPU** — The exporter only parses GPU 0 (matching the Python
  reference). Multi-GPU hosts will only see the first GPU.

## License

See the [LICENSE](LICENSE) file.
