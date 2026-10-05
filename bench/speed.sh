#!/bin/zsh
# speed.sh — idle-gated, interleaved speed arms for the port (and upstream torch-MPS as the reference arm).
#
# Every arm waits until the GPU is idle (5 consecutive AGX "Device Utilization %" samples < 10, read with the
# UNFILTERED ioreg form — `-k PerformanceStatistics` reads 0 under load) and the machine is thermally nominal, records
# both in the log, then runs in its own process. Arms are interleaved ABBA so session drift lands on both sides.
#
#   bench/speed.sh <outRoot> <arm>...      arm = <label>:<kind>:<framesDir>[:extra args]
#     kind = mlx-auto | mlx-kernel | mlx-dense | mlx-fp32 | torch | engine
#
# Env: THERM (a binary printing ProcessInfo.thermalState), FLASHVSR_MIRROR + TORCH_PY (torch arm), CLIP (engine arm
# input), LANE_BF16 / LANE_FP32 (lane dirs), SPIKE (mlxengine-forge/Tools/flashvsr-spike).
set -u
OUT=$1; shift
mkdir -p $OUT
B=${B:-.build-release/out/Products/Release/flashvsr-smoke}
LOG=$OUT/speed.log

gate() {
  local ok=0 u th
  while true; do
    u=$(ioreg -r -d 1 -c AGXAccelerator | grep -oE '"Device Utilization %"=[0-9]+' | head -1 | grep -oE '[0-9]+$')
    th=$($THERM)
    if [[ ${u:-100} -lt 10 && $th == nominal ]]; then ok=$((ok + 1)); else ok=0; fi
    [[ $ok -ge 5 ]] && break
    sleep 2
  done
  echo "$(date +%H:%M:%S) gate passed: gpu ${u}% thermal ${th}" >> $LOG
}

for arm in "$@"; do
  label=${arm%%:*}; rest=${arm#*:}; kind=${rest%%:*}; rest=${rest#*:}; frames=${rest%%:*}
  extra=""; [[ $rest == *:* ]] && extra=${rest#*:}
  gate
  d=$OUT/$label; rm -rf $d
  echo "$(date +%H:%M:%S) start $label ($kind $frames $extra)" >> $LOG
  case $kind in
    mlx-auto)   FLASHVSR_ATTN=auto FLASHVSR_PROFILE=1 $B run $frames $d $LANE_BF16 --dtype bf16 --lq-dtype bf16 \
                  --dec-dtype bf16 --cache-mb -1 ${=extra} > $d.log 2>&1 ;;
    mlx-kernel) FLASHVSR_ATTN=kernel FLASHVSR_PROFILE=1 $B run $frames $d $LANE_BF16 --dtype bf16 --lq-dtype bf16 \
                  --dec-dtype bf16 --cache-mb -1 ${=extra} > $d.log 2>&1 ;;
    mlx-dense)  FLASHVSR_ATTN=dense FLASHVSR_PROFILE=1 $B run $frames $d $LANE_BF16 --dtype bf16 --lq-dtype bf16 \
                  --dec-dtype bf16 --cache-mb -1 ${=extra} > $d.log 2>&1 ;;
    mlx-fp32)   FLASHVSR_ATTN=kernel FLASHVSR_PROFILE=1 $B run $frames $d $LANE_FP32 --cache-mb -1 ${=extra} > $d.log 2>&1 ;;
    torch)      FLASHVSR_MIRROR=$FLASHVSR_MIRROR $TORCH_PY $SPIKE/run_flashvsr.py $frames $d --dtype bf16 ${=extra} \
                  > $d.log 2>&1 ;;
    engine)     $B engine $frames $d.mp4 $LANE_BF16 ${=extra} > $d.log 2>&1 ;;
  esac
  echo "$(date +%H:%M:%S) end $label rc=$? gpu_after $(ioreg -r -d 1 -c AGXAccelerator | grep -oE '"Device Utilization %"=[0-9]+' | head -1) thermal_after $($THERM)" >> $LOG
done
