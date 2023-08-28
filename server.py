import uvicorn
import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

def get_input_ids(messages):
    input_ids = role_id['assistant']
    for message in reversed(messages):
        k,v = next(iter(message.items()))
        temp_ids = role_id[k]
        token_ids = tokenizer.encode(v)
        temp_ids.extend(token_ids)
        temp_ids.extend(input_ids)
        input_ids = temp_ids
        if len(input_ids) >= max_input_tokens:
            break
    return input_ids[-max_input_tokens:]
    
@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    messages = request_dict.pop("messages")
    stream = request_dict.pop("stream", False)
    sampling = sampling_params_default.copy()
    sampling.update(request_dict.get("sampling",{}))
    sampling_params = SamplingParams(**sampling)
    request_id = random_uuid()
    input_ids = get_input_ids(messages)
    results_generator = engine.generate(None,sampling_params,request_id,prompt_token_ids=input_ids)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.trust_remote_code = True
    # engine_args= AsyncEngineArgs(model = model,
                                 # trust_remote_code = True,
                                 # disable_log_stats = True,
                                 # disable_log_requests = True)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model)

    max_new_tokens = generation_config.max_new_tokens
    max_input_tokens = max(tokenizer.model_max_length // 2, tokenizer.model_max_length - max_new_tokens)
    role_id = {'user':[generation_config.user_token_id],'assistant':[generation_config.assistant_token_id]}
    sampling_params_default = dict(frequency_penalty=0.7,
                                             top_k =generation_config.top_k,
                                             top_p = generation_config.top_p, 
                                             temperature = generation_config.temperature,
                                             max_tokens = generation_config.max_new_tokens)
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
