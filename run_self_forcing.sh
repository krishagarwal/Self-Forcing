DIST_NNODES=8
echo "$HOSTNAME $(hostname -I)"
echo "$HOSTNAME $(hostname -I | awk '{print $2}')"
if [ "${HOSTNAME##*-}" -eq 0 ]; then
    ray start --head --node-ip-address=$(hostname -I | awk '{print $2}')
    until [ "$(ray status | grep node_ | wc -l | awk '{print $1}')" -eq $DIST_NNODES ]; do
        echo "waiting for all workers up..."
        sleep 10
    done
else
    until ray status; do
        sleep 3
        ray start --address="${HOSTNAME%-*}-0":6379 --node-ip-address=$(hostname -I | awk '{print $2}') --block
        sleep 3
    done
fi
echo "Ray all worker nodes started"
torchrun --nproc_per_node=8 --rdzv-conf="timeout=7200,read_timeout=7200,join_timeout=7200" train.py --config_path configs/self_forcing_sid.yaml --logdir /code-fsx/beidchen-sandbox/self_forcing_logs/self_forcing_dmd --wandb_name dmd_regular
