#!/bin/bash
# Run each test file in its OWN pytest process.
#
# Every file holds its engine in a session-scoped fixture, so a single pytest
# session keeps all of them resident at once - fp16 for test_parity, fp32+fp16
# for test_batch_parity, fp32 for test_continuous_parity and test_paged_parity.
# That is ~5 model copies on a machine that holds about two, and the run swaps
# itself to a standstill (observed: 48MB RSS, 11% CPU, 25 minutes, no progress,
# swap at 6.9GB of 8GB). Separate processes free each engine before the next.
cd "$(dirname "$0")"
fail=0
for f in tests/test_scheduler.py tests/test_block_manager.py tests/test_prefix_cache.py \
         tests/test_parity.py tests/test_paged_parity.py \
         tests/test_continuous_parity.py tests/test_batch_parity.py; do
  printf '%-36s ' "$(basename "$f")"
  out=$(./.venv/bin/python -m pytest "$f" -q 2>&1 | tail -2 | head -1)
  echo "$out"
  echo "$out" | grep -q "failed\|error" && fail=1
done
exit $fail
