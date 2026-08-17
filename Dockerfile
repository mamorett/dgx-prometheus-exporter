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
