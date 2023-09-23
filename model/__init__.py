from transformers import LlamaConfig
from .llama.pipeline_llama import LlamaForCausalLMPipe,LlamaCrossEntropyLoss
from .baichuan.modeling_baichuan import BaichuanConfig
from .baichuan.pipeline_baichuan import BaichuanForCausalLMPipe,BaichuanCrossEntropyLoss
from .qwen.modeling_qwen import QWenConfig
from .qwen.pipeline_qwen import QWenForCausalLMPipe,QWenCrossEntropyLoss

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