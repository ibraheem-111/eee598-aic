#!/usr/bin/env bash
# Stop parallel simulator environment $1 (started by run_sim.sh with AIC_ENV=$1) without touching the others.
I="${1:?env index}"; F="/tmp/aic_sim_env$I.pgid"
[ -f "$F" ] || exit 0
G="$(cat "$F")"
kill -INT -- "-$G" 2>/dev/null; sleep 5
kill -9 -- "-$G" 2>/dev/null
rm -f "$F"
