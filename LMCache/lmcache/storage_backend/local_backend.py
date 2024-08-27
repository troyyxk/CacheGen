from typing import Tuple, Optional, Iterator
import re
import io
import torch
import redis

from lmcache.utils import CacheEngineKey, KVCache
from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LMCLocalBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu memory.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig,
            local_storage_size: int
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        self.chunk_size = config.chunk_size 
        self.config = config
        self.dict = {}
        self.keys = []
        self.local_storage_size = local_storage_size

    def contains(
            self, 
            key: str,
        ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def put(
            self, 
            key: str,
            kv_chunk: str,
            blocking: bool = True,
        ) -> Optional[Tuple]:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        print("### [Local]Try put key {0}, value {1}".format(key, kv_chunk))
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")
        kv_to_send = None
        if len(self.keys) >= self.local_storage_size:
            evict_key = self.keys.pop(0)
            evict_value = self.dict[evict_key]
            del self.dict[evict_key]
            kv_to_send = (evict_key, evict_value)
        self.dict[key] = kv_chunk
        self.keys.append(key)
        if kv_to_send is not None:
            print("### [Local]Evict key {0}, value {1}".format(kv_to_send[0], kv_to_send[1]))
        return kv_to_send


    @_lmcache_nvtx_annotate
    def get(
            self,
            key: str,
        ) -> Optional[str]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        print("### [Local]Try get key {0}".format(key))
        if key in self.keys:
            self.keys.remove(key)
            self.keys.append(key)
        return self.dict.get(key, None)
