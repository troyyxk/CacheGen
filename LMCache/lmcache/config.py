from dataclasses import dataclass
import yaml
import re
from typing import Optional

@dataclass
class LMCacheEngineMetadata:
    ''' name of the LLM model '''
    model_name: str

    ''' world size when running under a distributed setting '''
    world_size: int

    ''' worker id when running under a distributed setting '''
    worker_id: int

    ''' the format of kv tensors '''
    fmt: str

@dataclass
class LMCacheEngineConfig:
    chunk_size: int
    local_device: str
    # remote_url: str
    remote_urls: list[str]
    num_url: int
    remote_serde: str # Can be "torch" or "cachegen"

    pipelined_backend: bool

    def from_defaults(
            chunk_size: int = 256,
            local_device: str = "cuda",
            remote_url: str = "redis://localhost:6379",
            remote_serde: str = "torch",
            pipelined_backend: bool = False,
        ) -> 'LMCacheEngineConfig':
        return LMCacheEngineConfig(
                chunk_size, local_device, [remote_url], 1, remote_serde,
                pipelined_backend)

    def from_legacy(
            chunk_size: int = 256,
            backend: str = "cuda",
            persist_path: str = None,
            remote_serde: str = "torch",
            pipelined_backend: bool = False,
        ) -> 'LMCacheEngineConfig':
        match backend:
            case "cpu" | "cuda":
                local_device = backend
                remote_urls = None
                num_url = 0
            # case url if re.match(r"(.*)://(.*):(\d+)", url):
            case _:
                local_device = None
                remote_urls = url.split(";")
                num_url = len(remote_urls)
        return LMCacheEngineConfig(
                chunk_size, local_device, remote_urls, num_url, remote_serde,
                pipelined_backend)

    @staticmethod
    def from_file(file_path: str) -> 'LMCacheEngineConfig':
        """
        Load the config from a yaml file
        """
        with open(file_path, 'r') as fin:
            config = yaml.safe_load(fin)

        chunk_size = config.get("chunk_size", 256)
        local_device = config.get("local_device", None)
        remote_urls = config.get("remote_url", None).split(";")
        num_url = len(remote_urls)
        remote_serde = config.get("remote_serde", "torch")
        pipelined_backend = config.get("pipelined_backend", False)
        return LMCacheEngineConfig(
                chunk_size, local_device, remote_urls, num_url, remote_serde,
                pipelined_backend)


### SOME GLOBAL CONFIGS 
# TODO: it needs to be manually updated in the code here, but cannot be really configured
class GlobalConfig:
    enable_debug: bool = True

    @classmethod
    def set_debug(cls, enable: bool):
        cls.enable_debug = enable

    @classmethod
    def is_debug(cls) -> bool:
        return cls.enable_debug
