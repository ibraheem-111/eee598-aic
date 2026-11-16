#!/bin/bash
# AIC simulator entrypoint for parallel environments, run inside the aic_eval container by run_sim.sh when
# AIC_ENV is set. Same as the image's /entrypoint.sh (no ACL), but environment i gets its own Zenoh router
# port (7447 + 10 i), ROS domain (i) and Gazebo partition, so several simulators share one container
# without seeing each other. Host processes join environment i by sourcing scripts/aic_env.sh.
set -e
. /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
I="${AIC_ENV:?AIC_ENV}"
PORT=$((7447 + 10 * I))
export ROS_DOMAIN_ID="$I" GZ_PARTITION="aic_env_$I"
export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5

export ZENOH_CONFIG_OVERRIDE="mode=\"router\";listen/endpoints=[\"tcp/[::]:$PORT\"];connect/endpoints=[];routing/router/peers_failover_brokering=true;transport/shared_memory/enabled=false"
echo "env $I: zenoh router on $PORT, ROS_DOMAIN_ID=$I, GZ_PARTITION=$GZ_PARTITION"
ros2 run rmw_zenoh_cpp rmw_zenohd &

export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/localhost:$PORT\"];transport/shared_memory/enabled=false"
exec ros2 launch aic_bringup aic_gz_bringup.launch.py "$@"
