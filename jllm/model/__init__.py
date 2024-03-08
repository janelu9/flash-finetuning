from .llama.pipeline_llama import LlamaForCausalLMPipe
from .baichuan.pipeline_baichuan import BaichuanForCausalLMPipe
from .qwen.pipeline_qwen import QWenForCausalLMPipe,QWenForClassCausalLMPipe
from .qwen2.pipeline_qwen2 import Qwen2ForCausalLMPipe
from .lora import convert_linear_layer_to_lora,_z3_params_to_fetch,convert_lora_to_linear_layer,only_optimize_lora_parameters
from .llama.parallel_llama import LlamaForCausalLMPipe as LlamaForCausalLMParallel
from .qwen2.parallel_qwen2 import Qwen2ForCausalLMPipe as Qwen2ForCausalLMParallel

ModelPipe = {
    'LlamaForCausalLM':LlamaForCausalLMPipe,
    'QWenLMHeadModel':QWenForCausalLMPipe,
    'Qwen2ForCausalLM':Qwen2ForCausalLMPipe,
    'QWenForClassCausalLM':QWenForClassCausalLMPipe,
    'BaichuanForCausalLM':BaichuanForCausalLMPipe
}
    
ModelParallel = {
    'LlamaForCausalLM':LlamaForCausalLMParallel,
    'Qwen2ForCausalLM':Qwen2ForCausalLMParallel
}
