from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
import json
from src.utils import *
import numpy as np
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward

# main.py
model, tokenizer = define_model_and_tokenizer("mistral-community/Mistral-7B-v0.2", num_gpus=1, max_gpu_memory=48)
data = load_testcases(DATASET_TO_PATH["longchat"])
text = data[1]['prompt']
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
generated = model.generate(input_ids, max_new_tokens=1, return_dict_in_generate=True)
kv = generated['past_key_values']
# rest and save tensor in tmp/mistral7b_longchat_data/raw_kv_0.pt

# run_cachegen.py
# data = load_testcases(DATASET_TO_PATH["longchat"])
kv_tokens = []
layer_to_device_id = {}
kv = pickle.load(open(f"./tmp/mistral7b_longchat_data/raw_kv_0.pkl", "rb"))
for i in range(len(kv)):
    layer_to_device_id[i] = kv[i][0].device.index

# 1st for loop
key_value = torch.load(f"./tmp/mistral7b_longchat_data/raw_kv_0.pt")
lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
bytes = cachegen_serializer.to_bytes(key_value)
pickle.dump(bytes, open(f"./tmp/encoded/0.pkl", "wb"))
kv_tokens += [key_value.shape[-2]]
decoded_kvs = []
average_acc = []

# 2nd for loop
os.environ['DOC_ID'] = str(0)
lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_tokens[0])
meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
deserializer = CacheGenDeserializer(lmcache_config, meta_data)
bytes = pickle.load(open(f"./tmp/encoded/0.pkl", "rb"))
decoded_kv = deserializer.from_bytes(bytes)
decoded_kvs += [decoded_kv.cpu()]

model, tokenizer = define_model_and_tokenizer("mistral-community/Mistral-7B-v0.2", num_gpus=1, max_gpu_memory=48)

# 3 rd for loop
decoded_kv = decoded_kvs[0].cuda()
decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
text = data[0]['prompt']
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(prediction, data[0]['label'][0])
