# EasyLLM

Running Large Language Model easily, faster and low-cost.

## Installation

```shell
pip install jllm
```

## Data Compression

This step is optional but recommended especially when your data are too big to be loaded to CPU memory at once.

### Conversion

Convert the raw data to token ids stored in parquet files.

```shell
python -m jllm.convert_raw_to_ids \
    --tokenizer baichuan-inc/Baichuan-13B-Chat \
    -i dataset0.jsonl \
    -o dataset0_Baichuan-13B-Chat
```

***Note**: Samples of pre-train dataset should be separated by `'\n\n'` in text files and be the value of  field `'text'` in jsonl files. Fine-tune dataset's format should be `[{'system':content},{'user':content},{'assistant':content},...] ` in jsonl files, field `'system'` is not necessary.*

### Shuffle

If you have multiple datasets, you shouldn't skip this step. It could shuffle all the datasets globally by rows like [Spark](https://spark.apache.org) doing. Firstly, move all the datasets stored with parquet folders into one directory. such as `datasets`:

```shell
datasets
├── dataset0_Baichuan-13B-Chat
│   ├── dataset0-00000
│   │   ├── dataset0-00000-00000.gzip.parquet
│   │   └── dataset0-00000-00001.gzip.parquet
│   └── dataset0-00001
│       ├── dataset0-00001-00000.gzip.parquet
│       └── dataset0-00001-00001.gzip.parquet
└── dataset1_Baichuan-13B-Chat
    ├── dataset1-00000
    │   ├── dataset1-00000-00000.gzip.parquet
    │   └── dataset1-00000-00001.gzip.parquet
    └── dataset1-00001
        ├── dataset1-00001-00000.gzip.parquet
        └── dataset1-00001-00001.gzip.parquet
```

Then run the following command to shuffle the rows in each dataset:

```shell
python -m jllm.shuffle_partition -d datasets --output shuffled_datasets
```

Every dataset would be shuffled and placed in `shuffled_datasets` with new partitions:

```shell
shuffled_datasets/
├── dataset0_Baichuan-13B-Chat-00000
│   ├── dataset0_Baichuan-13B-Chat-00000-00000.gzip.parquet
│   ├── dataset0_Baichuan-13B-Chat-00000-00001.gzip.parquet
│   ├── dataset0_Baichuan-13B-Chat-00000-00002.gzip.parquet
│   └── dataset0_Baichuan-13B-Chat-00000-00003.gzip.parquet
└── dataset1_Baichuan-13B-Chat-00000
    ├── dataset1_Baichuan-13B-Chat-00000-00000.gzip.parquet
    ├── dataset1_Baichuan-13B-Chat-00000-00001.gzip.parquet
    ├── dataset1_Baichuan-13B-Chat-00000-00002.gzip.parquet
    └── dataset1_Baichuan-13B-Chat-00000-00003.gzip.parquet
```

### Repartition 

Optional but recommended. 1B token ids in parquet files take up to 2G of hard disk at most but require approximately 10G of CPU memory. Setting `num_partition` according to the CPU memory of each worker.

```shell
num_partition=4 && ./repartition.sh shuffled_datasets $num_partition
```

You will get:

```shell
shuffled_datasets/
├── 5984729befe338e6a7-part-00000
│   ├── dataset0_Baichuan-13B-Chat-00000-00000.gzip.parquet
│   └── dataset1_Baichuan-13B-Chat-00000-00000.gzip.parquet
├── 5984729befe338e6a7-part-00001
│   ├── dataset0_Baichuan-13B-Chat-00000-00001.gzip.parquet
│   └── dataset1_Baichuan-13B-Chat-00000-00001.gzip.parquet
├── 5984729befe338e6a7-part-00002
│   ├── dataset0_Baichuan-13B-Chat-00000-00002.gzip.parquet
│   └── dataset1_Baichuan-13B-Chat-00000-00002.gzip.parquet
├── 5984729befe338e6a7-part-00003
│   ├── dataset0_Baichuan-13B-Chat-00000-00003.gzip.parquet
│   └── dataset1_Baichuan-13B-Chat-00000-00003.gzip.parquet
└── data.info
```

***Note**: You can also use Spark to shuffle the data if you have and want.*

## Model Training

Here are two training examples.

### ZERO

```shell
deepspeed train_zero.py \
    --model openlm-research/open_llama_13b \
    --train-data dataset0.jsonl
```

### Pipeline Parallelism

```shell
deepspeed --module jllm.train_pipe \
    --model baichuan-inc/Baichuan-13B-Chat \
    --train-data shuffled_datasets \
    --checkpoint_dir checkpoint
```

Generally, every GPU process reads one piece of data, that means one worker with 8 GPUs will need to allocate a total of 8x CPU memory for data.  But now they need just 1x if these GPUs belong to one pipeline under my special optimizations in this project . **I strongly recommend you to train your model with faster and low-cost Pipeline Parallelism** rather than ZERO. Pipeline engine could directly load and save model's weights in HuggingFace's format. It could also resume from the checkpoint. If you want to resume interruption, any configs related to training shouldn't be modified, otherwise it may load model's parameters only.

#### Checkpoint Conversion

Convert model's weights in checkpoint to HF format.

```shell
deepspeed --module jllm.convert_ckpt_to_hf \
	--ckpt checkpoint \
	--output_dir Baichuan-13B-Chat-Finetune
```

#### Supported Models

|    Model     | Pipeline Stages | Training Speed (tokens/s) |
| :----------: | :-------------: | :-----------------------: |
|  llama-13b   |        8        |         82540.54          |
| baichuan-13b |        8        |         67174.40          |
|   qwen-7b    |        4        |         122033.10         |
|   qwen-14b   |        8        |         75915.26          |

***Note**: The training speed of each model was measured on 64 NVIDIA A100-PCIE-40GB GPUs with data type of bfloat16 and batch token size of 4M(`batch_size*seq_length`).*

## Batch Inference

```shell
python batch_infer.py \
    --model Baichuan-13B-Chat-Finetune \
    --prompt-file prompt.txt
```

## API Server

Start the server:

```shell
python server.py --model Baichuan-13B-Chat-Finetune
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
