model_adapters: List[BaseModelAdapter] = []

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)


model, tokenizer = adapter.load_compress_model(
    model_path="mistral-community/Mistral-7B-v0.2",
    device="cuda",
    torch_dtype=torch.float16,
    revision="main",
)
def load_compress_model(model_path, device, torch_dtype, use_fast, revision="main"):
    # partially load model
    # `use_fast=True`` is not supported for some models.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast, revision=revision, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("mistral-community/Mistral-7B-v0.2", True, "main", True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=~use_fast, revision=revision, trust_remote_code=True
        )
    with init_empty_weights():
        # `trust_remote_code` should be set as `True` for both AutoConfig and AutoModel
        config = AutoConfig.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            revision=revision,
        )
        # some models are loaded by AutoModel but not AutoModelForCausalLM,
        # such as chatglm, chatglm2
        try:
            # google/flan-* models are based on an AutoModelForSeq2SeqLM.
            if "T5Config" in str(type(config)):
                model = AutoModelForSeq2SeqLM.from_config(
                    config, trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except NameError:
            model = AutoModel.from_config(config, trust_remote_code=True)
        linear_weights = get_compressed_list(model)
    if os.path.exists(model_path):
        # `model_path` is a local folder
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")
    else:
        # `model_path` is a cached Hugging Face repo
        # We don't necessarily need to download the model' repo again if there is a cache.
        # So check the default huggingface cache first.
        model_path_temp = os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub",
            "models--" + model_path.replace("/", "--"),
            "snapshots/",
        )
        downloaded = False
        if os.path.exists(model_path_temp):
            temp_last_dir = os.listdir(model_path_temp)[-1]
            model_path_temp = os.path.join(model_path_temp, temp_last_dir)
            base_pattern = os.path.join(model_path_temp, "pytorch_model*.bin")
            files = glob.glob(base_pattern)
            if len(files) > 0:
                downloaded = True

        if downloaded:
            model_path = model_path_temp
        else:
            model_path = snapshot_download(model_path, revision=revision)
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")

    files = glob.glob(base_pattern)
    use_safetensors = False
    if len(files) == 0:
        base_pattern = os.path.join(model_path, "*.safetensors")
        files = glob.glob(base_pattern)
        use_safetensors = True
    if len(files) == 0:
        raise ValueError(
            f"Cannot find any model weight files. "
            f"Please check your (cached) weight path: {model_path}"
        )

    compressed_state_dict = {}
    if use_safetensors:
        from safetensors.torch import load_file
    for filename in tqdm(files):
        if use_safetensors:
            tmp_state_dict = load_file(filename)
        else:
            tmp_state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage
            )
        for name in tmp_state_dict:
            if name in linear_weights:
                tensor = tmp_state_dict[name].to(device, dtype=torch_dtype)
                compressed_state_dict[name] = compress(
                    tensor, default_compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(
                    device, dtype=torch_dtype
                )
            tmp_state_dict[name] = None
            tensor = None
            gc.collect()
            torch.cuda.empty_cache()
            if device == "xpu":
                torch.xpu.empty_cache()
            if device == "npu":
                torch.npu.empty_cache()

    for name in model.state_dict():
        if name not in linear_weights:
            set_module_tensor_to_device(
                model, name, device, value=compressed_state_dict[name]
            )
    apply_compressed_weight(model, compressed_state_dict, device)

    if torch_dtype == torch.float16:
        model.half()
    model.to(device)
    model.eval()

    return model, tokenizer

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        return model, tokenizer

model, tokenizer = adapter.load_compress_model(
    model_path="mistral-community/Mistral-7B-v0.2",
    device="cuda",
    torch_dtype=torch.float16,
    revision="main",
)
    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

class MistralAdapter(BaseModelAdapter):
    """The model adapter for Mistral AI models"""

    def match(self, model_path: str):
        return "mistral" in model_path.lower() or "mixtral" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("mistral")

def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")

def load_model(
    model_path: str,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    exllama_config: Optional[ExllamaConfig] = None,
    xft_config: Optional[XftConfig] = None,
    revision: str = "main",
    debug: bool = False,
):
    """Load a model from Hugging Face."""
    import accelerate

    # get model adapter
    adapter = get_model_adapter(model_path)

    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
        if CPU_ISA in ["avx512_bf16", "amx"]:
            try:
                import intel_extension_for_pytorch as ipex

                kwargs = {"torch_dtype": torch.bfloat16}
            except ImportError:
                warnings.warn(
                    "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                )
    # TODO, USE cuda
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        import transformers

        version = tuple(int(v) for v in transformers.__version__.split("."))
        if version < (4, 35, 0):
            # NOTE: Recent transformers library seems to fix the mps issue, also
            # it has made some changes causing compatibility issues with our
            # original patch. So we only apply the patch for older versions.

            # Avoid bugs in mps backend by not using in-place operations.
            replace_llama_attn_with_non_inplace_operations()
    elif device == "xpu":
        kwargs = {"torch_dtype": torch.bfloat16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            warnings.warn(
                "Intel Extension for PyTorch is not installed, but is required for xpu inference."
            )
    elif device == "npu":
        kwargs = {"torch_dtype": torch.float16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import torch_npu
        except ImportError:
            warnings.warn("Ascend Extension for PyTorch is not installed.")
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # TODO, not here
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        # TODO, here!
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )
            model, tokenizer = adapter.load_compress_model(
                model_path="mistral-community/Mistral-7B-v0.2",
                device="cuda",
                torch_dtype=torch.float16,
                revision="main",
            )
        kwargs = {"torch_dtype": torch.float16}

            if debug:
                print(model)
            return model, tokenizer
    elif awq_config and awq_config.wbits < 16:
        assert (
            awq_config.wbits == 4
        ), "Currently we only support 4-bit inference for AWQ."
        model, tokenizer = load_awq_quantized(model_path, awq_config, device)
        if num_gpus != 1:
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=kwargs["max_memory"],
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                ],
            )
            model = accelerate.dispatch_model(
                model, device_map=device_map, offload_buffers=True
            )
        else:
            model.to(device)
        return model, tokenizer
    elif gptq_config and gptq_config.wbits < 16:
        model, tokenizer = load_gptq_quantized(model_path, gptq_config)
        if num_gpus != 1:
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=kwargs["max_memory"],
                no_split_module_classes=["LlamaDecoderLayer"],
            )
            model = accelerate.dispatch_model(
                model, device_map=device_map, offload_buffers=True
            )
        else:
            model.to(device)
        return model, tokenizer
    elif exllama_config:
        model, tokenizer = load_exllama_model(model_path, exllama_config)
        return model, tokenizer
    elif xft_config:
        model, tokenizer = load_xft_model(model_path, xft_config)
        return model, tokenizer
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype

    if os.environ.get("FASTCHAT_USE_MODELSCOPE", "False").lower() == "true":
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model_path):
                model_path = snapshot_download(model_id=model_path, revision=revision)
        except ImportError as e:
            warnings.warn(
                "Use model from www.modelscope.cn need pip install modelscope"
            )
            raise e

    # Load model
    model, tokenizer = adapter.load_model(model_path, kwargs)

    if (
        device == "cpu"
        and kwargs["torch_dtype"] is torch.bfloat16
        and CPU_ISA is not None
    ):
        model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device in (
        "mps",
        "xpu",
        "npu",
    ):
        model.to(device)

    if device == "xpu":
        model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    if debug:
        print(model)

    return model, tokenizer

model, tokenizer = load_model(
        model_path="mistral-community/Mistral-7B-v0.2",
        device="cuda",
        num_gpus=num_gpus,
        max_gpu_memory=f"{max_gpu_memory}GiB",
        load_8bit=True,
        cpu_offloading=False,
        debug=False,
    )
