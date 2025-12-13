import argparse

import torch

from modules.shifter.shift import Shift

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--key_shift", type=float, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    argument = parser.parse_args()

    Shift.shift_audio(
        input=argument.input,
        output=argument.output,
        key_shift=argument.key_shift,
        nsf_hifigan="checkpoints/nsf_hifigan/model",
        pitch_extractor="checkpoints/rmvpe/model.pt",
        device=argument.device,
        sample_rate=44100,
    )