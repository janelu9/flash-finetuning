# EasyLLM

Running Large Language Model easily and faster.

## Data Conversion

Convert the raw data to token ids stored in parquet format.

```shell
python convert_raws_to_ids.py -i news-commentary-v13-zh-en.txt
```

Do a repartition(optional but recommended).  The fewer partitions, the better shuffle, but the larger memory requirement during training. Setting `num_partition` according to the memory of each worker and the persistence of your cluster.

```shell
parquet_data_dir=news-commentary-v13-zh-en_open_llama_13b
num_partition=3
./repartition.sh $parquet_data_dir $num_partition
```



## Model Training

### ZERO ++

```shell
deepseed train_zero.py
```

### 3D Pipeline Parallelism

```shell
deepseed train_pipe.py
```

#### Attentions

If you want to switch to another model in the `models` folder,  modify these three places：

**Tokenizer**

```python
#convert_raws_to_ids.py:
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, fast_tokenizer = True, add_bos_token = True)
```

**Model config**

```python
#train_pipe.py:
config = LlamaConfig.from_pretrained(args.model_path)
```

 **Pipeline module**

```python
#train_pipe.py:
model = LlamaForCausalLMPipe(...
```

## API Server

Start the server:

```shell
python server.py --model openlm-research/open_llama_13b
```

Query the model :

```sehll
curl http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "sampling": {"max_tokens":32}
    }'
```

## Citation

If you find EasyLLM useful or use EasyLLM  code  in your research, please cite it in your publications.

```bibtex
@misc{EasyLLM,
  author       = {Jian Lu},
  title        = {EasyLLM: Running Large Language Model easily and faster},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/janelu9/EasyLLM.git}},
}
```

## Acknowledgment

This repository benefits from [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/HEAD/applications/DeepSpeed-Chat), [Flash-Attention](https://github.com/Dao-AILab/flash-attention.git) and [vLLM](https://github.com/vllm-project/vllm).
