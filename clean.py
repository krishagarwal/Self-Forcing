import glob
import os

folders = glob.glob("/code-fsx/beidchen-sandbox/self_forcing_logs/")

for folder in folders:
    checkpoint_folders = glob.glob(os.path.join(folder, "checkpoint_model_*"))
    if len(checkpoint_folders) == 0:
        continue
    checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("_")[-1]))
    # remove all but most recent
    for ckpt_folder in checkpoint_folders[:-1]:
        os.system(f"rm -rf {ckpt_folder}")

