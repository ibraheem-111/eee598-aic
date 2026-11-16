# Source to join parallel simulator environment $AIC_ENV (see scripts/sim_entrypoint.sh): Zenoh router port,
# ROS domain, Gazebo partition and SmolVLA server port. Does nothing when AIC_ENV is unset (single env).
if [ -n "${AIC_ENV:-}" ]; then
  export ROS_DOMAIN_ID="$AIC_ENV" GZ_PARTITION="aic_env_$AIC_ENV"
  export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/localhost:$((7447 + 10 * AIC_ENV))\"]"
  export AIC_SMOLVLA_PORT="$((50765 + AIC_ENV))"
fi
