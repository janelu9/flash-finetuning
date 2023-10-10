import argparse
from vllm import LLM,SamplingParams
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baichuan-inc/Baichuan-13B-Chat")
    parser.add_argument("--prompt_file", type=str,help='Prompt text file')
    parser.add_argument("--bacth_size", type=int, default=2)
    parser.add_argument("--output_file", type=str, default="result.txt")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model)
    sampling_params_default = SamplingParams(frequency_penalty=0.7,
                                             top_k =generation_config.top_k,
                                             top_p = generation_config.top_p, 
                                             temperature = generation_config.temperature,
                                             max_tokens = 32)#generation_config.max_new_tokens)
    role_id = {'user':[generation_config.user_token_id],'assistant':[generation_config.assistant_token_id]}
    model = LLM(args.model,trust_remote_code=True,gpu_memory_utilization=0.6)
    tmp_batch = []
    tmp_prompts = []
    with open(args.output_file,"w") as g:
        with open(args.prompt_file,"r") as f:
            prompt = f.readline()
            while prompt:
                prompt = prompt.strip()
                tmp_prompts.append(prompt)
                input_ids = role_id['user'] + tokenizer.encode(prompt) + role_id['assistant']
                tmp_batch.append(input_ids)
                if len(tmp_batch) == args.bacth_size:
                    output = model.generate(prompt_token_ids = tmp_batch,
                                            sampling_params = sampling_params_default,
                                            use_tqdm = False)
                    for prompt,answer in zip(tmp_prompts,output):
                        g.write(prompt + answer.outputs[0].text + "\n")
                    tmp_batch = []
                    tmp_prompts = []
                prompt = f.readline()
            if tmp_batch:
                    output = model.generate(prompt_token_ids = tmp_batch,
                                            sampling_params = sampling_params_default,
                                            use_tqdm = False)
                    for prompt,answer in zip(tmp_prompts,output):
                        g.write(prompt + answer.outputs[0].text + "\n")