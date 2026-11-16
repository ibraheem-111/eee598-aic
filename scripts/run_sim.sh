#!/usr/bin/env bash
# Start the AIC Gazebo simulator + scoring engine in the aic_eval container.
#   scripts/run_sim.sh [extra launch args...]
#   e.g. scripts/run_sim.sh gazebo_gui:=false launch_rviz:=false ground_truth:=true
# Uses docker directly (no sudo; you're in the docker group), as your user, so results land in
# ~/aic_results (or $AIC_RESULTS_DIR). `distrobox enter -r aic_eval -- /entrypoint.sh ...` does the same but asks for sudo.
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

# Parallel environments (AIC_ENV=i): our entrypoint with per-environment ports, in its own process group
# whose id goes to /tmp/aic_sim_env$i.pgid (the container shares the host's PID namespace, so
# scripts/aic_env_stop.sh can stop one environment without stopping the container).
if [ -n "${AIC_ENV:-}" ]; then
  exec docker exec -u "$(id -u)" -w "$HOME" \
    -e HOME="$HOME" -e DISPLAY="${DISPLAY:-:0}" -e XDG_RUNTIME_DIR="/run/user/$(id -u)" \
    -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -e AIC_RESULTS_DIR="${AIC_RESULTS_DIR:-$HOME/aic_results}" -e AIC_ENV="$AIC_ENV" \
    aic_eval setsid -w bash -c 'echo $$ > /tmp/aic_sim_env$AIC_ENV.pgid; exec "$0" "$@"' \
    "$(cd "$(dirname "$0")" && pwd)/sim_entrypoint.sh" ground_truth:=false start_aic_engine:=true "$@"
fi
exec docker exec -u "$(id -u)" -w "$HOME" \
  -e HOME="$HOME" -e DISPLAY="${DISPLAY:-:0}" -e XDG_RUNTIME_DIR="/run/user/$(id -u)" \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e AIC_RESULTS_DIR="${AIC_RESULTS_DIR:-$HOME/aic_results}" \
  aic_eval /entrypoint.sh ground_truth:=false start_aic_engine:=true "$@"
