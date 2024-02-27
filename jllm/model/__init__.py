from transformers import LlamaConfig,Qwen2Config
from .llama.pipeline_llama import LlamaForCausalLMPipe,LlamaCrossEntropyLoss
from .baichuan.modeling_baichuan import BaichuanConfig
from .baichuan.pipeline_baichuan import BaichuanForCausalLMPipe,BaichuanCrossEntropyLoss
from .qwen.modeling_qwen import QWenConfig
from .qwen.pipeline_qwen import QWenForCausalLMPipe,QWenCrossEntropyLoss,QWenMixedLoss,QWenForClassCausalLMPipe
from .qwen2.pipeline_qwen2 import Qwen2ForCausalLMPipe,Qwen2CrossEntropyLoss
from .lora import convert_linear_layer_to_lora,_z3_params_to_fetch,convert_lora_to_linear_layer,only_optimize_lora_parameters
from .llama.parallel_llama import (
    LlamaForCausalLMPipe as LlamaForCausalLMParallel,
    ParallelCrossEntropy
    )

ModelConfig = {
    'LlamaForCausalLM':LlamaConfig,
    'QWenLMHeadModel': QWenConfig,
    'QWenForClassCausalLM': QWenConfig,
    'BaichuanForCausalLM':BaichuanConfig,
    'Qwen2ForCausalLM':Qwen2Config
}

ModelPipe = {
    'LlamaForCausalLM':LlamaForCausalLMPipe,
    'QWenLMHeadModel':QWenForCausalLMPipe,
    'Qwen2ForCausalLM':Qwen2ForCausalLMPipe,
    'QWenForClassCausalLM':QWenForClassCausalLMPipe,
    'BaichuanForCausalLM':BaichuanForCausalLMPipe
}

CrossEntropyLossPipe = {
    'LlamaForCausalLM':LlamaCrossEntropyLoss,
    'QWenLMHeadModel': QWenCrossEntropyLoss,
    'Qwen2ForCausalLM': Qwen2CrossEntropyLoss,
    'QWenForClassCausalLM': QWenMixedLoss,
    'BaichuanForCausalLM':BaichuanCrossEntropyLoss
    }
    
ModelParallel = {
    'LlamaForCausalLM':LlamaForCausalLMParallel,
}

