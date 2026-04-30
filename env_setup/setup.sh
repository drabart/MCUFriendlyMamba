conda create -n mamba_314 python=3.14.4 -y
conda activate mamba_314

# install torch with AMD ROCm support
pip install -r torch_amd_install.txt

# install original mamba
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

pip install -r requirements.txt
