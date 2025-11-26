import argparse
# from model import CausalDiffusion
import torch
import time
import torch.distributed.checkpoint as dist_cp
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--exclude_ema", action="store_true")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# model = CausalDiffusion(config, device="cpu")

state = {}
print(f"Trying to load state dict from {args.ckpt_path}...")
start = time.time()
state = dist_cp.state_dict_loader._load_state_dict_from_keys(keys=None, checkpoint_id=args.ckpt_path)
end = time.time()
print(f"Loaded state dict in {end - start} seconds")

# orig_sd = model.generator.state_dict()
# for k, v in state["generator"].items():
#     assert state["generator"][k].shape == orig_sd[k].shape

if not args.exclude_ema:
    rename_param = (
        lambda name: name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )
    for k, v in state["generator_ema"].items():
        assert v.shape == orig_sd[rename_param(k)].shape, f"Shape mismatch for {k}: {v.shape} vs {orig_sd[rename_param(k)].shape}"
        assert torch.all(v == state["generator"][rename_param(k)]), f"Value mismatch for {k}"
# else:
#    del state["generator_ema"]

torch.save(state, args.output_path)
print(f"Saved converted checkpoint to {args.output_path}")
