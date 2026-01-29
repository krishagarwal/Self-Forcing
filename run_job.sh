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

# running on guparpit-nb-swe-ac (p5en)
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init_ema.yaml <= DONE
# kubectl apply -f yaml2/run_self_forcing_ema.yaml <= DONE
# kubectl apply -f yaml/run_monarch_oracle0.yaml <= DONE

# running on 2x ikiss-swe-skills-p5 2x shobhvas-runtime (p5)
# kubectl apply -f yaml/run_self_forcing_radial_attn.yaml <= DONE
# kubectl apply -f yaml/run_self_forcing_svg.yaml <= DONE
# kubectl apply -f yaml/run_self_forcing_radial_attn_nohack.yaml <= DONE
# kubectl apply -f yaml/run_self_forcing_svg_nohack.yaml <= DONE

# running on p5
# kubectl apply -f yaml/run_self_forcing_topk.yaml
# kubectl apply -f yaml/run_monarch_oracle1.yaml <= DONE
# kubectl apply -f yaml/run_monarch_oracle2.yaml <= DONE
# kubectl apply -f yaml/run_monarch_oracle3.yaml <= DONE

# kubectl apply -f yaml/run_monarch_oracle4.yaml <= DONE
# kubectl apply -f yaml/run_monarch_oracle5.yaml <= DONE
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init2_ema.yaml
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init3_ema.yaml <= DONE


# kubectl apply -f yaml2/run_wan_finetune_regular_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame2_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame3_ema.yaml

# evaluations
# kubectl apply -f yaml3/run_monarch_fast_framewise_dense_init_ema.yaml
# kubectl apply -f yaml3/run_self_forcing_ema.yaml
# kubectl apply -f yaml3/run_self_forcing_svg_nohack.yaml
# kubectl apply -f yaml3/run_self_forcing_svg.yaml
# kubectl apply -f yaml3/run_self_forcing_radial_attn_nohack.yaml
# kubectl apply -f yaml3/run_self_forcing_radial_attn.yaml


############## NEW RUNS ###############

# kubectl apply -f yaml/run_monarch_oracle0.yaml
# kubectl apply -f yaml/run_monarch_oracle1.yaml
# kubectl apply -f yaml/run_monarch_oracle2.yaml
# kubectl apply -f yaml/run_self_forcing_topk.yaml

# kubectl apply -f yaml/run_monarch_oracle0_exact.yaml
# kubectl apply -f yaml/run_monarch_oracle1_exact.yaml
# kubectl apply -f yaml/run_monarch_oracle2_exact.yaml

# kubectl apply -f yaml2/run_wan_finetune_regular_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame_ema.yaml
# kubectl apply -f yaml2/run_self_forcing_ema.yaml
# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init_ema.yaml

# kubectl apply -f yaml2/run_monarch_fast_framewise_dense_init3_ema.yaml
# kubectl apply -f yaml2/run_wan_finetune_monarch_fast_tied_frame2_ema.yaml

# kubectl apply -f yaml/run_wan_finetune_radial_attn.yaml
# kubectl apply -f yaml/run_wan_finetune_svg.yaml
# kubectl apply -f yaml/run_wan_finetune_topk.yaml

# kubectl apply -f yaml/run_monarch_oracle0.yaml
# kubectl apply -f yaml/run_monarch_oracle1.yaml
# kubectl apply -f yaml/run_monarch_oracle3.yaml
# kubectl apply -f yaml/run_monarch_oracle4.yaml
# kubectl apply -f yaml/run_self_forcing_topk.yaml
# kubectl apply -f yaml/run_self_forcing_radial_attn.yaml

# kubectl apply -f yaml/run_wan_14b_causal_test.yaml
# kubectl apply -f yaml/run_wan_14b_causal.yaml
# kubectl apply -f yaml/run_wan_14b_causal_monarch.yaml

# kubectl apply -f yaml/run_regular_14b.yaml
# kubectl apply -f yaml/run_monarch_14b_from_monarch.yaml
# kubectl apply -f yaml/run_monarch_14b_from_regular.yaml
# kubectl apply -f yaml/run_regular_14b_test.yaml


# kubectl apply -f yaml/run_wan_fewstep_inference_monarch_fast_framewise.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_monarch_slow_framewise.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_monarch_fast_framewise_reduce2xh.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_monarch_slow_framewise_reduce2xh.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_radial_attn.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_topk.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_svg.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_svg2.yaml

# kubectl apply -f yaml/run_wan_fewstep_dmd.yaml
# kubectl apply -f yaml/run_wan_fewstep_dmd_monarch_fast_framewise.yaml
# kubectl apply -f yaml/run_wan_fewstep_dmd_monarch_fast_framewise_reduce2xh.yaml
# kubectl apply -f yaml/run_wan_fewstep_dmd_vsa.yaml
# kubectl apply -f yaml/run_wan_fewstep_dmd_radial_attn.yaml
# kubectl apply -f yaml/run_wan_fewstep_dmd_vsa95.yaml
# kubectl apply -f yaml/run_wan_fewstep_dmd_vsa90.yaml

kubectl apply -f yaml/run_self_forcing_inference_monarch_slow_framewise_reduce2xh.yaml
kubectl apply -f yaml/run_self_forcing_inference_monarch_fast_framewise_reduce2xh.yaml

# kubectl apply -f yaml/run_wan_fewstep_inference_svg2_0.85.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_svg2_0.90.yaml
# kubectl apply -f yaml/run_wan_fewstep_inference_svg2_0.95.yaml

# kubectl apply -f yaml4/run_wan_fewstep_inference_svg2_0.85.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_svg2_0.90.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_svg2_0.95.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_svg.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_topk.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_monarch_fast_framewise.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_monarch_slow_framewise.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_radial_attn.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_monarch_fast_framewise_reduce2xh.yaml
# kubectl apply -f yaml4/run_wan_fewstep_inference_monarch_slow_framewise_reduce2xh.yaml


kubectl apply -f yaml4/self_forcing_inference_svg.yaml
# kubectl apply -f yaml4/self_forcing_inference_topk.yaml
# kubectl apply -f yaml4/self_forcing_inference.yaml
kubectl apply -f yaml4/self_forcing_inference_monarch_fast_framewise.yaml
kubectl apply -f yaml4/self_forcing_inference_monarch_slow_framewise.yaml
# kubectl apply -f yaml4/self_forcing_inference_radial_attn.yaml
kubectl apply -f yaml4/self_forcing_inference_monarch_fast_framewise_reduce2xh.yaml
kubectl apply -f yaml4/self_forcing_inference_monarch_slow_framewise_reduce2xh.yaml
