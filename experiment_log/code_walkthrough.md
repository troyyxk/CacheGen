# search for where kv cache is made

## LMCache
lm_connector.py LMCServerConnector.set(self, key: str, obj: bytes)
obj是kvcache

__init__.py CreateConnector
制造出来的connector是LMCServerConnector

remote_backend.py LMCRemoteBackend
self.connection 是 LMCServerConnector
put_blocking(kv_chunk) self.connection.set(self._combine_key(key), bs)，bs是kv cache, kv_chunk 是 kv cache
put(key, kv_chunk) 使用了 put_blocking()
下一步看在哪里用了 LMCRemoteBackend.put()

__init()__.py CreateStorageBackend()
制造了 LMCRemoteBackend

cache_engine.py LMCacheEngine
LMCacheEngine self.engine_ 用了 CreateStorageBackend() 是 LMCRemoteBackend
self.engine_.batched_put() 用了 LMCRemoteBackend.put()
n_chunks = self.engine_.batched_put(
        ((
            self._make_key(chunk_hash, fmt), 
            self._tuple_kv_to_blob(kv_chunk)
        ) for chunk_hash, kv_chunk in chunk_hashes_and_kvs), 
        blocking=blocking
    )
self._tuple_kv_to_blob()
- Convert the nested tuple of kv tensors to a single big tensor with 2 extra dimensions
self._make_key()
- 输出CacheEngineKey
store()里的kv_tensors

cache_engine.py LMCacheEngineBuilder
engine = LMCacheEngine(config, metadata)

TODO, 后面在哪用了 LMCacheEngine 或者 LMCacheEngineBuilder

## vllm branch: dev/lmcache-integration
woker.py Worker.__init__()
LMCacheEngineBuilder.get_or_create("vllm", lmcache_config, lmcache_metadata)
这里给 LMCacheEngineBuilder._instances["vllm"] 创立了 LMCacheEngine

model_runner.py ModelRunner.__init__()
self.cache_engine = LMCacheEngineBuilder.get("vllm")
这里的 cache_engine 就是之前在 woker.py 里创造的 LMCacheEngine
self.lmcache_driver = LMCVLLMDriver(self.cache_engine, self.model_config, self.parallel_config, self.device)
这里又储存到 lmcache_driver LMCVLLMDriver 在 lmcache_vllm 里

## lmcache_vllm
driver.py LMCVLLMDriver.__init__()
self.cache_engine = cache_engine
cache_engine 是 LMCacheEngine
retrive_and_inject() 和 collect_kv_and_store()
里用到了 self.cache_engine
主要追踪哪里用了 LMCVLLMDriver.collect_kv_and_store()
里面用了 self.cache_engine.store()

## vllm branch: dev/lmcache-integration
model_runner.py execute_model()
用了 LMCVLLMDriver.collect_kv_and_store()

worker.py execute_model()
用了 ModelRunner.execute_model()
输入的 self.gpu_cache 是 kv cache
接下来查看 self.gpu_cache 从哪拿的，然后看在哪制作 delta tensor 的
_init_cache_engine() is where self.gpu_cache is inited

cache_engine.py CacheEngine.__init__()
CacheEngine 只管存储和读取，需要看看计算 kv cache的地方在哪里

woker.py Worker
self.cache_engine 是 CacheEngine
也只有这个地方用了 CacheEngine , 其他的都是它本身所在的地方和 test
除了 _init_cache_engine() , 就只有 cache_swap() 用了 CacheEngine
cache_swap() 在 execute_model() 里用了

# find where quantization level is used
## LMCache
cachegen_basics.py CacheGenConfig
from_model_name() 里面限定了 quantization level

cachegen_encoder.py CacheGenSerializer
cachegen_decoder.py CacheGenDeserializer
都用了 CacheGenConfig, self.cachegen_config 就是

__init__.py CreateSerde()
创造了 CacheGenSerializer 和CacheGenDeserializer
LMCRemoteBackend 用了 CacheGenSerializer
在 put_blocking() 对kv_chunk用了
bs = self.serializer.to_bytes(kv_chunk)

cachegen_encoder.py cachegen_encoder()

