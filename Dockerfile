# syntax=docker/dockerfile:1
FROM golang:1.21-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./

ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /dgx-exporter .

# The runtime stage must provide libc and the ELF interpreter. `scratch` cannot
# exec nvidia-smi even when the NVIDIA container runtime injects the binary and
# the /dev/nvidia* nodes: the kernel reports ENOENT for the missing interpreter,
# and the exporter then silently serves no GPU metrics at all.
FROM debian:bookworm-slim
COPY --from=build /dgx-exporter /usr/local/bin/dgx-exporter
EXPOSE 9273
ENV DGX_EXPORTER_PORT=9273 COLLECT_INTERVAL=10
ENTRYPOINT ["/usr/local/bin/dgx-exporter"]
