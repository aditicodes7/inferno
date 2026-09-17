#!/bin/bash
# Run each test file in its OWN pytest process.
#
# Every file holds its engine in a session-scoped fixture, so a single pytest
# session keeps all of them resident at once - fp16 for test_parity, fp32+fp16
# for test_batch_parity, fp32 for the continuous/paged/prefix parity files.
# That is ~5 model copies on a machine that holds about two, and the run swaps
# itself to a standstill (observed: 48MB RSS, 11% CPU, 25 minutes, no progress,
# swap at 6.9GB of 8GB). Separate processes free each engine before the next.
#
# The tensor-free files run first, so a logic regression surfaces in 0.05s
# rather than after minutes of model loading.
cd "$(dirname "$0")"

# Use the local venv when there is one (Mac dev), otherwise whatever python is
# on PATH (Kaggle, Colab, a rented box). Hardcoding ./.venv/bin/python meant
# this script could only ever run on the machine it was written on.
if [ -n "$PYTHON" ]; then :
elif [ -x ./.venv/bin/python ]; then PYTHON=./.venv/bin/python
elif command -v python3 >/dev/null; then PYTHON=python3
else PYTHON=python
fi
echo "python: $($PYTHON -c 'import sys;print(sys.version.split()[0], sys.executable)')"
echo

fail=0
for f in tests/test_scheduler.py tests/test_block_manager.py tests/test_prefix_cache.py \
         tests/test_parity.py tests/test_paged_parity.py tests/test_prefix_parity.py \
         tests/test_continuous_parity.py tests/test_batch_parity.py; do
  printf '%-32s ' "$(basename "$f")"
  # Match pytest's summary line explicitly. Do NOT use tail/head to guess at
  # it: files that print during tests (capsys.disabled) interleave with the
  # progress dots, and an earlier version of this script captured a bare "."
  # and then grepped THAT for failures - so a failing file would have been
  # reported green.
  out=$("$PYTHON" -m pytest "$f" -q 2>&1)
  line=$(echo "$out" | grep -E '^[0-9]+ (passed|failed)|[0-9]+ (passed|failed|error)' | tail -1)
  if echo "$out" | grep -qE '[0-9]+ (failed|error)'; then
    echo "FAIL  $line"; fail=1
  elif [ -z "$line" ]; then
    echo "NO SUMMARY LINE - treating as failure"; fail=1
  else
    echo "$line"
  fi
done
[ $fail -eq 0 ] && echo "ALL GREEN" || echo "FAILURES PRESENT"
exit $fail
