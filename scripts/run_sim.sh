#!/usr/bin/env bash
# Start the AIC Gazebo simulator + scoring engine in the aic_eval container.
#   scripts/run_sim.sh [extra launch args...]
#   e.g. scripts/run_sim.sh gazebo_gui:=false launch_rviz:=false ground_truth:=true
# Uses docker directly (no sudo; you're in the docker group), as your user, so results land in
# ~/aic_results. `distrobox enter -r aic_eval -- /entrypoint.sh ...` does the same but asks for sudo.
set -euo pipefail

# On every start, distrobox's init links the host's NVIDIA libraries into the container. If Gazebo
# launches before that finishes, it silently falls back to CPU rendering and runs at ~2% real time.
# So wait for init's "container_setup_done" marker from *this* start.
if [ "$(docker inspect -f '{{.State.Running}}' aic_eval)" != "true" ]; then
  before=$(docker logs aic_eval 2>&1 | grep -c '^container_setup_done' || true)
  docker start aic_eval >/dev/null
  for _ in $(seq 300); do
    now=$(docker logs aic_eval 2>&1 | grep -c '^container_setup_done' || true)
    [ "$now" -gt "$before" ] && break
    sleep 1
  done
  [ "$now" -gt "$before" ] || { echo "aic_eval init did not finish" >&2; exit 1; }
fi

exec docker exec -u "$(id -u)" -w "$HOME" \
  -e HOME="$HOME" -e DISPLAY="${DISPLAY:-:0}" -e XDG_RUNTIME_DIR="/run/user/$(id -u)" \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  aic_eval /entrypoint.sh ground_truth:=false start_aic_engine:=true "$@"
