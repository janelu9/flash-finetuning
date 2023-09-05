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

### 3D Pipeline Parallelism (recommended)

```shell
deepseed train_pipe.py
```

#### Attentions

If you want to switch to another model in the `model` folder,  modify these three places：

**Tokenizer**

```python
#convert_raws_to_ids.py:
tokenizer = BaichuanTokenizer.from_pretrained(args.tokenizer, ...
```

**Model config**

```python
#train_pipe.py:
config = BaichuanConfig.from_pretrained(args.model_path)
```

 **Pipeline module**

```python
#train_pipe.py:
model = BaichuanForCausalLMPipe(...
```

Pipeline engine could load and save model's weights with Hugging Face's format directly. It could also load and resume from the checkpoint. If you want to resume interruption, any configs shouldn't be modified.

## Batch Inference

```shell
python batch_infer.py --model baichuan-inc/Baichuan-13B-Chat --prompt-file prompt.txt
```

## API Server

Start the server:

```shell
python server.py --model baichuan-inc/Baichuan-13B-Chat
```

Query the model :

```sehll
curl http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "messages":[{"user": "San Francisco is a"}],
        "sampling":{"max_tokens":32}
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
