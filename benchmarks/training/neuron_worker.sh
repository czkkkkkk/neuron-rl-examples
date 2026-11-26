#!/usr/bin/env bash
set -euo pipefail
# torchrun's rank and process groups stay in their original logical order.
IFS=, read -r -a cores <<< "${QWEN38_NEURON_CORE_ORDER:?}"
export LOCAL_RANK=${cores[LOCAL_RANK]}
exec "$VERL_NEURON_PYTHON" "$@"
