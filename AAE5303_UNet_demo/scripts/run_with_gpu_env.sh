#!/usr/bin/env bash
set -euo pipefail

# Force PyTorch to load host driver libcuda first (avoids Error 804 in this env).
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1

exec "$@"
