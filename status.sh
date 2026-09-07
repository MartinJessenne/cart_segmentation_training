#!/usr/bin/env bash
# Live training status, run ON molab (no ssh hop).
#
#   ./status.sh          one snapshot
#   ./status.sh watch    refresh every 30s until Ctrl-C
#
# The first snapshot after a restart reports an AVERAGE rate, which torch.compile
# warmup and the first-epoch ramp drag down. Every later snapshot reports the
# rate measured since the previous one ("recent"), which is what the ETA uses.
set -uo pipefail

case "${1:-once}" in
  watch)
    while true; do
      clear
      python3 /root/progress.py
      printf '\n(refreshing every 30s — Ctrl-C to stop)\n'
      sleep 30
    done
    ;;
  *)
    python3 /root/progress.py
    ;;
esac
