#!/usr/bin/env bash
# Peak GPU memory (whole training process, incl. CUDA context) and step time of a 12-step lerobot-train run.
#   BENCH_DIR=<dir with .venv> scripts/bench_vla_train_memory.sh <name> <lerobot-train args...>
# Appends to $BENCH_DIR/results.tsv: name  status  peak_MiB  sec_per_step. Results: docs/vla_and_rl_choice.md
set -uo pipefail
cd "${BENCH_DIR:?set BENCH_DIR to a dir with a .venv that has lerobot[training,<policy extras>]}"
name=$1; shift
log=logs/$name.log; mkdir -p logs out
rm -rf out/$name
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
.venv/bin/lerobot-train --dataset.repo_id=lerobot/svla_so101_pickplace --output_dir=out/$name \
  --job_name=$name --steps=12 --log_freq=1 --save_checkpoint=false --wandb.enable=false \
  --num_workers=2 --policy.push_to_hub=false --policy.device=cuda "$@" >"$log" 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  pids=$(pgrep -P $pid; echo $pid)
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v p="$(echo $pids)" 'BEGIN{n=split(p,a," ");for(i=1;i<=n;i++)s[a[i]]=1} ($1 in s){t+=$2} END{print t+0}')
  [ "$m" -gt "$peak" ] && peak=$m
  sleep 0.5
done
wait $pid; rc=$?
status=ok; [ $rc -ne 0 ] && status="fail($rc)"
grep -qiE 'out of memory|OutOfMemoryError' "$log" && status=OOM
# lerobot logs "updt_s:<seconds>" per logged step
sps=$(grep -oE 'updt_s:[0-9.]+' "$log" | tail -8 | cut -d: -f2 | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "-"}')
printf "%s\t%s\t%s\t%s\n" "$name" "$status" "$peak" "$sps" | tee -a results.tsv
