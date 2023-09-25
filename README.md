# EasyLLM

Running Large Language Model easily, faster and low-cost.

## Data Compression

This step is optional but recommended especially when your data are too big to be loaded to CPU memory at once.

### Conversion

Convert the raw data to token ids stored in parquet files.

```shell
python convert_raw_to_ids.py \
    --tokenizer baichuan-inc/Baichuan-13B-Chat \
    -i news-commentary-v13-zh-en.txt \
    -o news-commentary-v13-zh-en_Baichuan-13B-Chat
```

### Repartition 

Optional but recommended. The fewer partitions, the better shuffle, but the larger CPU memory requirement during training. 1B token ids in parquet files take up to 2G of hard disk at most but require approximately 10G of CPU memory. Setting `num_partition` according to the CPU memory of each worker.

```shell
num_partition=2 && ./repartition.sh news-commentary-v13-zh-en_Baichuan-13B-Chat $num_partition
```

## Model Training

### ZERO

```shell
deepseed train_zero.py \
    --model openlm-research/open_llama_13b \
    --train-data news-commentary-v13-zh-en.txt
```

### Pipeline Parallelism

```shell
deepseed train_pipe.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --train-data news-commentary-v13-zh-en_Baichuan-13B-Chat
```

Generally, every GPU process reads one piece of data, that means one worker with 8 GPUs will need to allocate a total of 8x CPU memory for data.  But now they need just 1x if these GPUs belong to one pipeline under my special optimizations in this project . **I strongly recommend you to train your model with faster and low-cost Pipeline Parallelism** rather than ZERO. Pipeline engine could directly load and save model's weights in HuggingFace's format. It could also resume from the checkpoint. If you want to resume interruption, any configs related to training shouldn't be modified, otherwise it may load model's parameters only.

#### Supported Models

|    Model     | Pipeline Stages | Training Speed (tokens/s) |
| :----------: | :-------------: | :-----------------------: |
|  llama-13b   |        8        |         82391.04          |
| baichuan-13b |        8        |         67174.40          |
|   qwen-7b    |        4        |         122033.10         |

**Note**: The training speed of each model was measured on 64 NVIDIA A100-PCIE-40GB GPUs with data type of bfloat16 and batch token size of 4M(`batch_size*seq_length`).

## Batch Inference

```shell
python batch_infer.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --prompt-file prompt.txt
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
  title        = {EasyLLM: Running Large Language Model easily, faster and low-cost.},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/janelu9/EasyLLM.git}},
}
```

## Acknowledgment

This repository benefits from [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Flash-Attention](https://github.com/Dao-AILab/flash-attention.git), [xFormers](https://github.com/facebookresearch/xformers) and [vLLM](https://github.com/vllm-project/vllm).
