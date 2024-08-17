search for where kv cache is made

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

TODO, 后面在哪用了 LMCacheEngine 或者LMCacheEngineBuilder

## lmcache-vllm

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

