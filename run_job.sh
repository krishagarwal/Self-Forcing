#!/bin/bash
kubectl apply -f yaml/run_monarch_slow.yaml
kubectl apply -f yaml/run_monarch_fast.yaml
