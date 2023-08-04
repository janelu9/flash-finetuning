# EasyLLM

Running Large Language Model easily and faster.

## Data Conversion

Convert the raw data to token ids stored in parquet format.

```shell
python convert_raws_to_ids.py -i news-commentary-v13-zh-en.txt
```

Do a shuffle and repartition(optional but recommended).  The fewer partitions, the better, but the larger the memory requirement during training. Setting `$num_partition` according to your node's memory.

```shell
parquet_data_dir=news-commentary-v13-zh-en_parquet
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

This repository benefits from [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Flash-Attention](https://github.com/Dao-AILab/flash-attention.git) and [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/HEAD/applications/DeepSpeed-Chat). 
