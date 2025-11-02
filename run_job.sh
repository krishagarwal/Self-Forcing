#!/bin/bash
# kubectl apply -f yaml/run_monarch_slow_0.85.yaml
# kubectl apply -f yaml/run_monarch_fast_0.85.yaml
# kubectl apply -f yaml/run_monarch_slow_0.95.yaml
kubectl apply -f yaml/run_monarch_fast_0.95.yaml
# kubectl apply -f yaml/run_self_forcing.yaml
# kubectl apply -f yaml/run_wan_finetune_monarch_fast_0.85.yaml
# kubectl apply -f yaml/run_wan_finetune_monarch_fast_0.95.yaml
# kubectl apply -f yaml/run_wan_finetune_regular.yaml
kubectl apply -f yaml/run_wan_finetune_monarch_fast_tied_frame.yaml
kubectl apply -f yaml/run_wan_finetune_monarch_fast_cross_frame.yaml
kubectl apply -f yaml/run_wan_finetune_monarch_fast_cross_frame_reduced_w.yaml
kubectl apply -f yaml/run_wan_finetune_monarch_fast_cross_frame_reduced_h.yaml
