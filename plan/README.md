# motion_planning

## Installation

```bash
conda create -n mp python=3.9
conda activate mp
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2

export CUDA_HOME=/usr/local/cuda-11.8
git clone git@github.com:wrc042/TorchSDF.git
(cd TorchSDF; pip install -e .)

git clone git@github.com:mzhmxzh/torchprimitivesdf.git
(cd torchprimitivesdf; pip install -e .)

pip install plotly transforms3d open3d==0.17.0 urdf_parser_py tensorboard wandb coacd rich

git clone git@github.com:huggingface/diffusers.git
(cd diffusers; pip install -e ".[torch]")

pip install PyOpenGL glfw pyglm

pip install dm_control

sudo apt-get install castxml
pip install pygccxml pyplusplus
sudo apt install clang libode-dev

pip install numpy==1.26.4
unzip ompl-1.5.2.zip
echo "<path to ompl-1.5.2>/py-bindings" >> "<path to conda environment>"/lib/python3.9/site-packages/ompl.pth
pip install graspnetapi
pip install numpy==1.26.4
pip install transforms3d==0.4.2
pip install ikpy pin pybullet pycollada

# only needed by curobo
sudo apt install git-lfs
git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout v0.7.4
pip install -e . --no-build-isolation # this takes a long time (about 20 min)
# use python -m pytest . to test
pip install tabulate 

pip install meshcat
sudo apt install libzmq3-dev

```