conda create --name MFRNP python=3.10.11

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install numpy glob2 pyyaml wandb ray[tune] matplotlib