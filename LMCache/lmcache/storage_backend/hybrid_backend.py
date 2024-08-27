from typing import Tuple, Optional, Iterator, List
import re
import abc
import io
import torch
import redis
import time
import pickle
import queue
import threading
from multiprocessing import Process, Queue

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.remote_backend import LMCRemoteBackend, LMCPipelinedRemoteBackend
from lmcache.storage_backend.local_backend import LMCLocalBackend
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector
from lmcache.utils import _lmcache_nvtx_annotate, CacheEngineKey

logger = init_logger(__name__)

        
class LMCHybridBackend(LMCBackendInterface):
    """
    A hybrid backend that uses both local and remote backend to store and retrieve data.
    It implements write-through and read-through caching.
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata, local_storage_size: int):
        self.local_store = LMCLocalBackend(config, local_storage_size)
        if config.pipelined_backend:
            self.remote_store = LMCPipelinedRemoteBackend(config, metadata)
        else:
            self.remote_store = LMCRemoteBackend(config, metadata)
        
        # TODO add a configuration item to do this
        # self._prefetch(metadata)
           
    def _prefetch(
        self,
        metadata: LMCacheEngineMetadata
    ):
        keys = self.remote_store.list()
        nfetched = 0
        logger.info("Found %d keys in remote backend", len(keys))
        logger.debug(f"Metadata is {metadata}")
        start = time.perf_counter()
        for key in keys:
            if key.model_name != metadata.model_name or \
                    key.worker_id != metadata.worker_id or \
                    key.world_size != metadata.world_size:
                continue

            retrived_data = self.remote_store.get(key)
            if retrived_data is not None:
                self.local_store.put(key, retrived_data)
                nfetched += 1

        end = time.perf_counter()

        logger.info("Pre-fetched %d keys from remote backend, used %.2f sec", nfetched, end - start)
    
    def contains(
            self,
            key: Tuple[str, str],
        ) -> bool:
        return self.local_store.contains(key) or self.remote_store.contains(key)

    def put(
            self,
            key: str,
            value: str,
            blocking: bool = True,
        ):
        evict_kv = self.local_store.put(key, value, blocking = True)
        if evict_kv is not None:
            self.remote_store.put(evict_kv[0], evict_kv[1], blocking)
    
        
    @_lmcache_nvtx_annotate
    def get(
            self,
            key: str,
        ) -> Optional[str]:
        value = self.local_store.get(key)
        if value is None:
            value = self.remote_store.get(key)
            if value is not None:
                self.put(key, value)
        return value
    

    @_lmcache_nvtx_annotate
    def batched_get(
            self,
            keys: str,
        ) -> Iterator[Optional[str]]:
        ret = []
        remote_queries = []
        remote_query_idxs = []
        for idx, key in enumerate(keys):
            value = self.local_store.get(key)
            ret.append(value)
            if value is None:
                remote_queries.append(key)
                remote_query_idxs.append(idx)

        remote_query_results = self.remote_store.batched_get(remote_queries)
        for idx, key, result in zip(remote_query_idxs, 
                                    remote_queries, 
                                    remote_query_results):
            if result is not None:
                self.put(key, result)
                ret[idx] = result
        return ret

    def close(self):
        self.local_store.close()
        self.remote_store.close()
