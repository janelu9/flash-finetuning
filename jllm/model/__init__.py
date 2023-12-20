from transformers import LlamaConfig
from .llama.pipeline_llama import LlamaForCausalLMPipe,LlamaCrossEntropyLoss
from .baichuan.modeling_baichuan import BaichuanConfig
from .baichuan.pipeline_baichuan import BaichuanForCausalLMPipe,BaichuanCrossEntropyLoss
from .qwen.modeling_qwen import QWenConfig
from .qwen.pipeline_qwen import QWenForCausalLMPipe,QWenCrossEntropyLoss,QWenMixedLoss,QWenForClassCausalLMPipe
from .lora import convert_linear_layer_to_lora,_z3_params_to_fetch,convert_lora_to_linear_layer,only_optimize_lora_parameters
from .llama.parallel_llama import (
    LlamaForCausalLMPipe as LlamaForCausalLMParallel,
    ParallelCrossEntropy
    )

ModelConfig = {
    'LlamaForCausalLM':LlamaConfig,
    'QWenLMHeadModel': QWenConfig,
    'QWenForClassCausalLM': QWenConfig,
    'BaichuanForCausalLM':BaichuanConfig
}

ModelPipe = {
    'LlamaForCausalLM':LlamaForCausalLMPipe,
    'QWenLMHeadModel':QWenForCausalLMPipe,
    'QWenForClassCausalLM':QWenForClassCausalLMPipe,
    'BaichuanForCausalLM':BaichuanForCausalLMPipe
}

CrossEntropyLossPipe = {
    'LlamaForCausalLM':LlamaCrossEntropyLoss,
    'QWenLMHeadModel': QWenCrossEntropyLoss,
    'QWenForClassCausalLM': QWenMixedLoss,
    'BaichuanForCausalLM':BaichuanCrossEntropyLoss
    }
    
ModelParallel = {
    'LlamaForCausalLM':LlamaForCausalLMParallel,
}

