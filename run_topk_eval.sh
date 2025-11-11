#!/usr/bin/env bash

for gpu in {0..7}; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu

    start=$((gpu * 8))
    end=$(((gpu + 1) * 8))

    echo "GPU $gpu handling idx [$start, $end)"

    for topk in 0.9 1.0; do
      outdir=topk_eval/$topk
      mkdir -p "$outdir"

      export ATTN_TOPK_PCT=$topk
      python inference.py \
        --config_path configs/self_forcing_sid.yaml \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --data_path prompts/MovieGenVideoBench.txt \
        --output_folder "$outdir" \
        --use_ema \
        --num_output_frames 21 \
        --save_with_index \
        --idx_start "$start" \
        --idx_end "$end"
    done
  ) &
done

wait   # wait for all 8 GPUs’ jobs to finish
