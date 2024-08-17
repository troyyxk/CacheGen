# mirror
pip install pyzmq==26.0.3 --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install transformers==4.42.3 --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install cuda-python --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# problems
when running
```
    python setup.py install
```
meet this:
```
  File "/home/troy/miniconda3/envs/cachegen/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1987, in _get_cuda_arch_flags
    arch_list[-1] += '+PTX'
```
https://github.com/pytorch/extension-cpp/issues/71
problems is caused by no cudo card :(
cuda and pytorch does not match
torch.cuda.is_available() is false
how to match and set true with no gpu?

python TORCH_CUDA_ARCH_LIST="YOUR_GPUs_CC+PTX" setup.py install

# new approach
try alicloud server

(cachegen) root@autodl-container-690911b33c-d8271f5a:~/CacheGen/LMCache/third_party/torchac_cuda# history
    1  ll
    2  git clone https://github.com/troyyxk/CacheGen.git
    3  pwd
    4  ll
    5  cd CacheGen/
    6  pwd
    7  cd /root/CacheGen
    8  conda env create -f env.yaml
    9  conda activate cachegen
   10  conda init
   11  conda activate cachegen
   12  source activate base
   13  conda activate cachegen
   14  ll
   15  pip install -e LMCache
   16  cd /root/CacheGen/LMCache/third_party/torchac_cuda
   17  python setup.py install
   18  ll
   19  pwd
   20  history

go to source code to see how load_data() works

kv cache and limiatation
https://www.youtube.com/watch?v=z07GStMex4w
pageattention
https://www.youtube.com/watch?v=rQa0JPBG1Ps
https://www.youtube.com/watch?v=5ZlavKF_98U&t

when runing 7b.sh, got this error:
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like mistral-community/Mistral-7B-v0.2 is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
fixed by using:
export HF_ENDPOINT=https://hf-mirror.com
source:
https://github.com/huggingface/diffusers/issues/6223

conda info --envs get 

~/.cache/huggingface/hub/models--mistral-community--Mistral-7B-v0.2#

root@autodl-container-690911b33c-d8271f5a:~/.cache/huggingface/hub/models--mistral-community--Mistral-7B-v0.2# ll
.no_exist/
blobs/
refs/
snapshots/

have a lot of problem downloading the model, its a bug in the huggingface backend
https://github.com/huggingface/text-generation-inference/issues/1186

model-00002-of-00003.safetensors
79a5f871a0fbb7919f263d78caff0d4d3b7a1d444552db3011ada70f48fd5ee4.incomplete

autodl科学上网
https://zhuanlan.zhihu.com/p/685018159

where was CacheGenEncoderImpl been used?
where was 
