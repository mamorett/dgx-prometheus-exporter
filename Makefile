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
	docker build --build-arg VERSION=$(VERSION) -t dgx-exporter:$(VERSION) -t dgx-exporter:latest .

clean:
	rm -f $(BINARY)
