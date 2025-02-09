import os

ckpt_paths = [
    (
        # "path/to/VLA-Checkpoint-{epoch}-{steps}.ckpt",
        "RoboVLMs/checkpoints/kosmos_ph_calvin_abcd.pt",
        # "path/to/VLA-Checkpoint-config.json",
        "RoboVLMs/configs/kosmos_ph_calvin_abcd.json"
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/run_eval_raw_ddp_torchrun.sh {} {}".format(ckpt, config))
