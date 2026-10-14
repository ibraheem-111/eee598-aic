#!/usr/bin/env bash
# Run the SmolVLA policy in aic_model. Start scripts/run_smolvla_server.sh first.
# Settings are AIC_SMOLVLA_* environment variables, see policy/aic_smolvla/RunSmolVLA.py.
exec "$(dirname "$0")/run_policy.sh" aic_smolvla.RunSmolVLA
