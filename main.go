package main

import (
	"log"
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
	port := 9273
	if v := os.Getenv("DGX_EXPORTER_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			port = p
		} else {
			log.Printf("warning: bad DGX_EXPORTER_PORT %q, using default %d", v, port)
		}
	}
	interval := 10
	if v := os.Getenv("COLLECT_INTERVAL"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			interval = i
		} else {
			log.Printf("warning: bad COLLECT_INTERVAL %q, using default %d", v, interval)
		}
	}

	log.Printf("dgx-prometheus-exporter v%s starting (port=%d interval=%ds)", version, port, interval)

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
		p := r.URL.Path
		if p == "" || p == "/metrics" || (len(p) > 1 && strings.TrimRight(p, "/") == "/metrics") {
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

	if err := http.ListenAndServe("0.0.0.0:"+strconv.Itoa(port), mux); err != nil {
		log.Fatal(err)
	}
}
