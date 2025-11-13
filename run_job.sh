#!/bin/bash
# # kubectl apply -f yaml/run_monarch_slow_0.85.yaml
# # kubectl apply -f yaml/run_monarch_fast_0.85.yaml
# # kubectl apply -f yaml/run_monarch_slow_0.95.yaml
# # kubectl apply -f yaml/run_monarch_fast_0.95.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_reduced_h.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_reduced_w.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_reduced_dense_init.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_dense_init.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_reduced_framewise_init.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_framewise_init.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_reduced_lowsparse_init.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_lowsparse_init.yaml
# kubectl apply -f yaml/run_self_forcing.yaml
# kubectl apply -f yaml/run_self_forcing_test.yaml
# # kubectl apply -f yaml/run_wan_finetune_monarch_fast_0.85.yaml
# # kubectl apply -f yaml/run_wan_finetune_monarch_fast_0.95.yaml
# kubectl apply -f yaml/run_wan_finetune_regular.yaml
# kubectl apply -f yaml/run_wan_finetune_monarch_fast_tied_frame.yaml
# kubectl apply -f yaml/run_wan_finetune_monarch_fast_tied_frame2.yaml
# kubectl apply -f yaml/run_wan_finetune_monarch_fast_tied_frame3.yaml
# # kubectl apply -f yaml/run_wan_finetune_monarch_fast_cross_frame.yaml
# # kubectl apply -f yaml/run_wan_finetune_monarch_fast_cross_frame_reduced_w.yaml
# # kubectl apply -f yaml/run_wan_finetune_monarch_fast_cross_frame_reduced_h.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_sweep0.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_sweep1.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_sweep2.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_sweep3.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_sweep4.yaml
# # kubectl apply -f yaml/run_monarch_fast_crossframe_sweep5.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_h.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_w.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_max_sparse.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_max_sparse2.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_max_sparse3.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_max_sparse4.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_dense_init.yaml
# kubectl apply -f yaml/run_monarch_fast_framewise_dense_init3.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_reduced_lowsparse_init.yaml
# # kubectl apply -f yaml/run_monarch_fast_framewise_lowsparse_init.yaml
# # kubectl apply -f yaml/run_topk_eval.yaml
# kubectl apply -f yaml/run_monarch_fast_framewise_dense_init.yaml
# kubectl apply -f yaml/run_monarch_fast_framewise_dense_init2.yaml

# kubectl apply -f yaml/run_monarch_oracle0.yaml
# kubectl apply -f yaml/run_self_forcing_radial_attn.yaml
# kubectl apply -f yaml/run_self_forcing_svg.yaml
# kubectl apply -f yaml/run_self_forcing_topk.yaml
# kubectl apply -f yaml2/run_self_forcing_ema.yaml
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init_ema.yaml
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init2_ema.yaml
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init3_ema.yaml

kubectl apply -f yaml/run_self_forcing_radial_attn_nohack.yaml
kubectl apply -f yaml/run_self_forcing_svg_nohack.yaml
kubectl apply -f yaml/run_monarch_oracle1.yaml
kubectl apply -f yaml/run_monarch_oracle2.yaml

# kubectl apply -f yaml/run_monarch_oracle3.yaml
# kubectl apply -f yaml/run_monarch_oracle4.yaml
# kubectl apply -f yaml/run_monarch_oracle5.yaml


# kubectl apply -f yaml2/run_wan_finetune_regular_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame2_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame3_ema.yaml
