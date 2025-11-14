#!/bin/bash

# aws s3 cp s3://agi-mm-training-shared-us-east-2/beidchen/data/dmd_monarch_framewise_fast_denseinit_ema_vbench_videos /workspace/vbench_videos_sparse --region us-east-2 --recursive
# aws s3 cp s3://agi-mm-training-shared-us-east-2/beidchen/data/dmd_regular_ema_vbench_videos /workspace/vbench_videos_dense --region us-east-2 --recursive
mkdir -p /workspace/split_videos
python split.py --num_groups 4 --max_videos 200 --full_folder /workspace/vbench_videos_dense --sparse_folder /workspace/vbench_videos_sparse --output_base /workspace/split_videos/regular_vs_denseinit --prompt_dir .
