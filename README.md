# EasyLLM

Running Large Language Model easily.

## Data Conversion

Convert the raw data to token ids stored in parquet format.

```shell
python convert_raws_to_ids.py -i news-commentary-v13-zh-en.txt
```

## Model Trainning

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
#convert_raws_to_ids.py: line 118
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, fast_tokenizer = True, add_bos_token = True)
```

**Model config**

```python
#train_pipe.py: line 140
config = LlamaConfig.from_pretrained(args.model_path)
```

 **Pipeline module**

```python
#train_pipe.py: line 146
model = LlamaForCausalLMPipe(...)
```

## Citation

If you find EasyLLM useful or use EasyLLM  code  in your research, please cite it in your publications.

```bibtex
@misc{EasyLLM,
  author       = {Jian Lu},
  title        = {EasyLLM: Running Large Language Model easily},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/janelu9/EasyLLM.git}},
}
```

## Acknowledgment

This repository benefits from [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai), [DeepSpeed](https://github.com/microsoft/DeepSpeed), and [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/HEAD/applications/DeepSpeed-Chat). 

