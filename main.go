package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// version is stamped at build time via -ldflags "-X main.version=...".
var version = "dev"

func main() {
	defaultPort := 9273
	if v := os.Getenv("DGX_EXPORTER_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			defaultPort = p
		} else {
			log.Printf("warning: bad DGX_EXPORTER_PORT %q, using default %d", v, defaultPort)
		}
	}
	defaultInterval := 10
	if v := os.Getenv("COLLECT_INTERVAL"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			defaultInterval = i
		} else {
			log.Printf("warning: bad COLLECT_INTERVAL %q, using default %d", v, defaultInterval)
		}
	}

	var (
		port        int
		interval    int
		listenHost  string
		showVersion bool
	)

	flag.IntVar(&port, "port", defaultPort, "HTTP listen port (env: DGX_EXPORTER_PORT)")
	flag.IntVar(&port, "p", defaultPort, "HTTP listen port (shorthand)")
	flag.IntVar(&interval, "interval", defaultInterval, "Collection interval in seconds (env: COLLECT_INTERVAL)")
	flag.IntVar(&interval, "i", defaultInterval, "Collection interval in seconds (shorthand)")
	flag.StringVar(&listenHost, "addr", "0.0.0.0", "Listen address/host")
	flag.StringVar(&listenHost, "a", "0.0.0.0", "Listen address/host (shorthand)")
	flag.BoolVar(&showVersion, "version", false, "Print version and exit")
	flag.BoolVar(&showVersion, "v", false, "Print version and exit (shorthand)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s (v%s):\n\n", os.Args[0], version)
		fmt.Fprintf(os.Stderr, "Options:\n")
		fmt.Fprintf(os.Stderr, "  -p, --port int\n")
		fmt.Fprintf(os.Stderr, "    \tHTTP listen port (default %d, env: DGX_EXPORTER_PORT)\n", defaultPort)
		fmt.Fprintf(os.Stderr, "  -i, --interval int\n")
		fmt.Fprintf(os.Stderr, "    \tCollection interval in seconds (default %d, env: COLLECT_INTERVAL)\n", defaultInterval)
		fmt.Fprintf(os.Stderr, "  -a, --addr string\n")
		fmt.Fprintf(os.Stderr, "    \tListen address (default \"0.0.0.0\")\n")
		fmt.Fprintf(os.Stderr, "  -v, --version\n")
		fmt.Fprintf(os.Stderr, "    \tPrint version and exit\n")
		fmt.Fprintf(os.Stderr, "  -h, --help\n")
		fmt.Fprintf(os.Stderr, "    \tPrint this help message and exit\n")
	}

	flag.Parse()

	if showVersion {
		fmt.Printf("dgx-prometheus-exporter v%s\n", version)
		os.Exit(0)
	}

	log.Printf("dgx-prometheus-exporter v%s starting (%s:%d interval=%ds)", version, listenHost, port, interval)

	c := newCollector("")
	latest := &atomic.Pointer[string]{}
	starting := "# dgx custom collector starting...\n"
	latest.Store(&starting)

	go func() {
		for {
			text, err := c.collect()
			if err != nil {
				payload := errorPayload(err)
				latest.Store(&payload)
			} else {
				latest.Store(&text)
			}
			time.Sleep(time.Duration(interval) * time.Second)
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimRight(r.URL.Path, "/")
		if p == "" || p == "/metrics" {
			s := latest.Load()
			body := []byte(*s)
			w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(200)
			w.Write(body)
			return
		}
		http.NotFound(w, r)
	})

	addr := net.JoinHostPort(listenHost, strconv.Itoa(port))
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
