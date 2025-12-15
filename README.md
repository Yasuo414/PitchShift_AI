# PitchShift AI
## Introduction
PitchShift AI is a small, practically insignificant utility that uses AI to raise or lower the pitch in recordings. <br>

## How to get started
We will need Python 3.9 or newer.<br>
For clarity and to prevent chaos with Python environments, I recommend Anaconda3 or its lighter version, Miniconda3.<br>
Both can be found at (https://www.anaconda.com/download/success)[https://www.anaconda.com/download/success]. Install them according to their guides.<br>

Then create a new Python 3.9 or higher environment using the command<br>
`conda create --name PitchShift_AI python=3.9 (or higher) --yes`<br>
You may need to agree to their TOS. If prompted, accept them.<br>
After successfully creating the environment, activate it using<br>
`conda activate PitchShift_AI`<br>
Let's install PyTorch before the dependencies,<br>
because if we installed them in reverse order,<br>
the dependencies would overwrite the PyTorch we need, which could cause problems/errors.<br>
`pip3 install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130`<br>
Now we can install dependencies<br>
`pip3 install -r requirements.txt`<br>

## Usage/Inference
