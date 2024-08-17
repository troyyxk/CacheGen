# for vmware pro
# cd /home/troy/Desktop
# # python
# sudo apt install python3-pip
# # git
# sudo apt install git
# # get conda
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# ~/miniconda3/bin/conda init bash
# ~/miniconda3/bin/conda init zsh
# # get git repo
# git clone https://github.com/troyyxk/CacheGen.git
# # install conda dependencies
# cd /home/troy/Desktop/CacheGen
# conda env create -f env.yaml
# conda activate cachegen
# pip install -e LMCache
# cd /home/troy/Desktop/CacheGen/LMCache/third_party/torchac_cuda
# python setup.py install

# for AutoDL
# check the data disk path
source ~/.bashrc
# enable global network access
# source: https://www.autodl.com/docs/network_turbo/
source /etc/network_turbo
# get git repo
git clone https://github.com/troyyxk/CacheGen.git
# install conda dependencies
cd /root/autodl-tmp/CacheGen
conda env create -f env.yaml
# active base environment
source activate base
conda activate cachegen
pip install -e LMCache
cd /root/autodl-tmp/CacheGen/LMCache/third_party/torchac_cuda
python setup.py install
# start 7b experiment
cd /root/autodl-tmp/CacheGen
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/autodl-tmp/huggingface
export SAVE_DIR=./tmp/
bash scripts/7b.sh longchat 0

