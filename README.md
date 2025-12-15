# PitchShift AI
## Introduction
PitchShift AI is a small, practically insignificant utility that uses AI to raise or lower the pitch in recordings. 

## How to get started
We will need Python 3.9 or newer.
For clarity and to prevent chaos with Python environments, I recommend Anaconda3 or its lighter version, Miniconda3.
Both can be found at (https://www.anaconda.com/download/success)[https://www.anaconda.com/download/success]. Install them according to their guides.

Then create a new Python 3.9 or higher environment using the command
`conda create --name PitchShift_AI python=3.9 (or higher) --yes`
You may need to agree to their TOS. If prompted, accept them.
After successfully creating the environment, activate it using
`conda activate PitchShift_AI`
Let's install PyTorch before the dependencies,
because if we installed them in reverse order,
the dependencies would overwrite the PyTorch we need, which could cause problems/errors.
`pip3 install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130`
Now we can install dependencies
`pip3 install -r requirements.txt`

## Usage/Inference
