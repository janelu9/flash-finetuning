# EasyLLM

Running Large Language Model easily, faster and low-cost.

## Installation

```shell
pip install jllm
```

## Data Handling

This step is optional but recommended especially when your data are too big to be loaded to CPU memory at once.

### Conversion

Convert the raw data to token ids stored in parquet files.

```shell
python -m jllm.raw2ids \
    --tokenizer baichuan-inc/Baichuan-13B-Chat \
    -i dataset0.jsonl \
    -o dataset0_Baichuan-13B-Chat
```

***Note**: Samples of pre-train dataset should be separated by `'\n\n'` in text files or be the value of  key`'text'` in jsonl files. Fine-tune dataset's format should be `[{'system':content},{'user':content},{'assistant':content},...] ` in each row of jsonl files, key`'system'` is not necessary.*

### Shuffle

If you have multiple datasets, you shouldn't skip this step. It could shuffle all the datasets globally by rows like [Spark](https://spark.apache.org) doing. 

Firstly, move all the datasets stored in parquet folders into one directory. such as `datasets`:

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

Then run the following command to shuffle the rows inner each dataset and distribute them to new blocks, `num_block` is recommended to be the multiple of next step's repartition number.

```shell
python -m jllm.shuffle_datasets -d datasets -o shuffled_datasets -n 4
```

Every dataset would be shuffled and placed in `shuffled_datasets` with several times of `num_block` parquet files:

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
python -m jllm.repartition -d shuffled_datasets -n 4
```

The datasets will be:

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
deepspeed -H $HOSTFILE \
    --train_zero.py \
    --model openlm-research/open_llama_13b \
    --train_data dataset0.jsonl
```

### Pipeline Parallelism

```shell
deepspeed -H $HOSTFILE \
    --module jllm.train_pipe \
    --model baichuan-inc/Baichuan-13B-Chat \
    --train_data shuffled_datasets \
    --pipe_parallel_size 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --ds_config ds_config.py \
    --checkpoint checkpoint \
    --max_num_checkpoints 2
```

***Note**: Arguments `train_data` and `eval_data` also support `jsonl` file. Run `python -m jllm.train_pipe -h ` for more arguments.* 

Generally, every GPU process reads one piece of data, that means one node with 8 GPUs will need to allocate a total of 8x CPU memory for data.  But now they need just 1x if these GPUs belong to one pipeline under my special optimizations in this project . **I strongly recommend you to train your model with faster and low-cost Pipeline Parallelism** rather than ZERO. Pipeline engine could directly load and save model's weights in HuggingFace's format. It could also load weights from checkpoint. If you want to resume interruption, any configs related to training shouldn't be modified. 

The engine was designed saving checkpoint background at default to save more time for training. That means **you should set the docker container's '`shm-size`' as 30% of a node's total GPU memory at least** including the spaces for each process's optimizer states. **Don't save checkpoint too frequently** unless you disable checkpoint background via the argument '`--not_ckpt_background`', otherwise your SHM may be out of memory.

#### Checkpoint Conversion

Convert model's weights in checkpoint to HF format.

```shell
deepspeed --module jllm.ckpt2hf \
	--model baichuan-inc/Baichuan-13B-Chat \
	--pipe_parallel_size 8 \
	--ckpt checkpoint \
	--hf Baichuan-13B-Chat-Finetune
```

If your model don't have any `lora` weights, you can also convert the checkpoint without GPUs by:

```shell
python -m jllm.nolora_ckpt2hf \
	--model baichuan-inc/Baichuan-13B-Chat \
	--ckpt checkpoint \
	--hf Baichuan-13B-Chat-Finetune
```

#### Supported Models

|    Model     | Training Speed (tokens/s) |
| :----------: | :-----------------------: |
|  llama-13b   |         92749.82          |
| baichuan-13b |         79765.50          |
|   qwen-14b   |         80749.57          |

***Note**: The training speed of each model was measured on 64 NVIDIA A100-PCIE-40GB GPUs linked by 100Gb/s bandwidth of InfiniBand with data type of bfloat16 and batch token size of 2048\*2048 (batch_size\*sequence_length,  batch_size = micro_batch_size \* gradient_accumulation_steps).*

|  Model   | Training Speed (tokens/s) |
| :------: | :-----------------------: |
| llama-7b |         21200.90          |

***Note**: Measured on 8 NVIDIA A100-PCIE-40GB GPUs with data type of bfloat16 and batch token size of 128\*2048.*

## Inference

vLLm is quoted here for Inference.

### Batch Inference

```shell
python batch_infer.py \
    --model Baichuan-13B-Chat-Finetune \
    --prompt_file prompt.txt
```

### API Server

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

This repository benefits from [DeepSpeed](https://github.com/microsoft/DeepSpeed),  [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) and [Flash-Attention](https://github.com/Dao-AILab/flash-attention.git).
