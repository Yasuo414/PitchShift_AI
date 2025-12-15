# PitchShift AI
## Introduction
PitchShift AI is a small, practically insignificant utility that uses AI to raise or lower the pitch in recordings. <br>

## How to get started
The following commands need to be executed in the conda environment of python 3.9 or newer
```
# Install PyTorch related core dependencies, skip if installed
# Reference: https://pytorch.org/get-started/locally/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install the project dependencies
pip3 install -r requirements.txt
```

## Download checkpoints
I use NSF HiFiGAN and RMVPE, so... that's about it.
NSF-HiFiGAN:
- [FishAudio](https://github.com/fishaudio/fish-diffusion/releases/download/v2.0.0/nsf_hifigan-stable-v1.zip)
- [OpenVPI](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)<br>
TIP: I recommend using FishAudio NSF HiFiGAN rather than the OpenVPI version, as FishAudio has a wider pitch range.

RMVPE:
- [yxllc's Version](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)
- [lj1995's Version](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)<br>
TIP: I recommend the yxlllc version, as it has a wider range again.

Place both checkpoints in this structure:
```
<PitchShift AI folder>
|---- checkpoints
|      |---- nsf_hifigan
|             |---- model
|             |---- config.json
|      |---- rmvpe
|             |---- model.pt
|---- modules
|     | #....
```

## How to use
Find the 44.1kHz recording you want to edit.
Then enter this command:
```
python shift.py --input <your_WAV_file/your_folder> --output ./<output_WAV_file/output_folder> --key_shift <how many semitones you want to shift> --device <run it on CUDA or CPU> --recursive (optional) --overwrite (optional) --format (wav, flac, mp3) --verbose (if debug) --quiet (optional) --silent (optional) --add_suffix (if you want to see the shifted value in output filename)
```
