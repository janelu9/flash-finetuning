from transformers import LlamaConfig
from .llama.pipeline_llama import LlamaForCausalLMPipe,LlamaCrossEntropyLoss
from .baichuan.modeling_baichuan import BaichuanConfig
from .baichuan.pipeline_baichuan import BaichuanForCausalLMPipe,BaichuanCrossEntropyLoss
from .qwen.modeling_qwen import QWenConfig
from .qwen.pipeline_qwen import QWenForCausalLMPipe,QWenCrossEntropyLoss
from .lora import convert_linear_layer_to_lora,_z3_params_to_fetch,convert_lora_to_linear_layer,only_optimize_lora_parameters

ModelConfig = {
    'llama':LlamaConfig,
    'qwen': QWenConfig,
    'baichuan':BaichuanConfig
}

ModelPipe = {
    'llama':LlamaForCausalLMPipe,
    'qwen': QWenForCausalLMPipe,
    'baichuan':BaichuanForCausalLMPipe
}

CrossEntropyLossPipe = {
    'llama':LlamaCrossEntropyLoss,
    'qwen': QWenCrossEntropyLoss,
    'baichuan':BaichuanCrossEntropyLoss
    }