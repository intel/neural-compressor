#!/bin/bash

# Shared helpers for benchmark scripts.

die() { echo "[ERROR] $*" >&2; exit 1; }

init_benchmark_paths() {
    BENCHMARK_DIR="${BENCHMARK_DIR:-$PWD}"
    AGENT_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os/mini-swe-agent"
    AGENT_DIR_VERIFIED="${BENCHMARK_DIR}/mini-swe-agent"
    MCP_DIR="${BENCHMARK_DIR}/mcp-atlas"
    LOG_DIR="${BENCHMARK_DIR}/logs"
}
