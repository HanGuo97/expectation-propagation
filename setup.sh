pip install --upgrade pip
# Some of the requirements have conflict dependencies, notably on `jax`.
# So we will first install these dependencies, and overwrite them with
# the `jax` versions we need below.
pip install -r requirements.txt

# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Weight and Bias
pip install -U wandb
wandb login
