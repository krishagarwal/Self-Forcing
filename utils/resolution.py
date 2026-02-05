# TODO: quick patch to get resolution info

import os

resolution = os.getenv("RESOLUTION", "480p")

if resolution == "480p":
    latent_shape = [1, 21, 16, 60, 104]
    seq_shape = [21, 30, 52]
    frame_height = 30
    frame_width = 52
    total_seq_len = 21 * 30 * 52
    print("Initialized to 480p resolution.")
elif resolution == "720p":
    latent_shape = [1, 21, 16, 90, 160]
    seq_shape = [21, 45, 80]
    frame_height = 45
    frame_width = 80
    total_seq_len = 21 * 45 * 80
    print("Initialized to 720p resolution.")
elif resolution == "1080p":
    latent_shape = [1, 21, 16, 134, 240]
    seq_shape = [21, 67, 120]
    frame_height = 67
    frame_width = 120
    total_seq_len = 21 * 67 * 120
    print("Initialized to 1080p resolution.")
