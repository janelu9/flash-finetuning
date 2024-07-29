# EasyLLM

Running Large Language Model easily, faster and low-cost. 

Make PCIE as fast as NVlinks.

## Installation

```shell
pip wheel -e . --no-deps && pip install jllm-*-py3-none-any.whl
```

## Data Handling

This step is optional but recommended especially when your data are too big to be loaded to CPU memory at once.

### Conversion

Convert the raw data to token ids stored in parquet files.

```shell
python -m jllm.raw2ids \
    --tokenizer Qwen1.5-14B-Chat \
    -i dataset0.jsonl \
    -o dataset0_Qwen1.5-14B-Chat
```

***Note**: Samples of pre-train dataset should be separated by `'\n\n'` in text files or be the value of  key`'text'` in jsonl files. Fine-tune dataset's format should be `[{'system':content},{'user':content},{'assistant':content},...] ` in each row of jsonl files, key`'system'` is not necessary.*

For Vision Language Model:

```shell
python -m jllm.raw2ids \
    --tokenizer InternVL2-8B \
    -i dataset_vl.jsonl \
    --image_path images \
    --sep
```

Folder `images` stores all the images data.  Format of  `dataset_vl.jsonl` is like:

`[{'user':['Give a description of these pictures please.\n <image>....','image0.jpg',...]},{'assistant':'This is ....'}]`

`--sep` indicates saving images into parquet separately, less disk spaces will be occupied and patches will be generated during training.

### Shuffle

If you have multiple datasets, you shouldn't skip this step. It could shuffle all the datasets globally by rows like [Spark](https://spark.apache.org) doing. 

Firstly, move all the datasets stored in parquet folders into one directory. such as `datasets`:

```shell
datasets
├── dataset0_Qwen1.5-14B-Chat
│   ├── dataset0-00000
│   │   ├── dataset0-00000-00000.gzip.parquet
│   │   └── dataset0-00000-00001.gzip.parquet
│   └── dataset0-00001
│       ├── dataset0-00001-00000.gzip.parquet
│       └── dataset0-00001-00001.gzip.parquet
└── dataset1_Qwen1.5-14B-Chat
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
├── dataset0_Qwen1.5-14B-Chat-00000
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00000.gzip.parquet
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00001.gzip.parquet
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00002.gzip.parquet
│   └── dataset0_Qwen1.5-14B-Chat-00000-00003.gzip.parquet
└── dataset1_Qwen1.5-14B-Chat-00000
    ├── dataset1_Qwen1.5-14B-Chat-00000-00000.gzip.parquet
    ├── dataset1_Qwen1.5-14B-Chat-00000-00001.gzip.parquet
    ├── dataset1_Qwen1.5-14B-Chat-00000-00002.gzip.parquet
    └── dataset1_Qwen1.5-14B-Chat-00000-00003.gzip.parquet
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
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00000.gzip.parquet
│   └── dataset1_Qwen1.5-14B-Chat-00000-00000.gzip.parquet
├── 5984729befe338e6a7-part-00001
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00001.gzip.parquet
│   └── dataset1_Qwen1.5-14B-Chat-00000-00001.gzip.parquet
├── 5984729befe338e6a7-part-00002
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00002.gzip.parquet
│   └── dataset1_Qwen1.5-14B-Chat-00000-00002.gzip.parquet
├── 5984729befe338e6a7-part-00003
│   ├── dataset0_Qwen1.5-14B-Chat-00000-00003.gzip.parquet
│   └── dataset1_Qwen1.5-14B-Chat-00000-00003.gzip.parquet
└── data.info
```

***Note**: You can also use Spark to shuffle the data if you have and want.*

## Model Training

Here are two training examples.

### ZERO

```shell
deepspeed -H $HOSTFILE \
    --train_zero.py \
    --model Qwen1.5-14B-Chat \
    --train_data dataset0.jsonl
```

Both GPU and NPU are supported.

### 3D Parallelism

```shell
deepspeed -H $HOSTFILE \
    --module jllm.train_pipe \
    --model Qwen1.5-14B-Chat \
    --train_data shuffled_datasets \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --ds_config ds_config.py \
    --checkpoint checkpoint \
    --max_num_checkpoints 2 \
    --partition_method fast \
    --split_dlayer \
    --checkpoint_interval 2 
```

***Note**: Arguments `train_data` and `eval_data` also support `jsonl` file. Run `python -m jllm.train_pipe -h ` for more arguments.* 

Generally, every GPU process reads one piece of data, that means one node with 8 GPUs will need to allocate a total of 8x CPU memory for data.  But now they need just 1x if these GPUs belong to one pipeline under my special optimizations in this project . **I strongly recommend you to train your model with faster and low-cost Pipeline Parallelism** rather than ZERO. Pipeline engine could directly load and save model's weights in HuggingFace's format. It could also load weights from checkpoint. If you want to resume interruption, any configs related to training shouldn't be modified. 

The engine was designed to save checkpoint through background process by default to save more time for training. **Don't save checkpoint too frequently** unless you disable checkpoint in background via the argument '`--background_executor none`' to avoid out of CPU memory.

Setting `partition_method` to be`fast` will always get a faster training when memory are enough.

#### Checkpoint Conversion

Convert model's weights in checkpoint to HF format.

```shell
deepspeed --module jllm.ckpt2hf \
	--model Qwen1.5-14B-Chat \
	--pipe_parallel_size 8 \
	--ckpt checkpoint \
	--hf Qwen1.5-14B-Chat-Finetune
```

If your model don't have any `lora` weights, you can also convert the checkpoint without GPUs by:

```shell
python -m jllm.nolora_ckpt2hf \
	--model Qwen1.5-14B-Chat \
	--ckpt checkpoint \
	--hf Qwen1.5-14B-Chat-Finetune
```

#### Supported Models

|    Model     | Training Speed (tokens/s) |
| :----------: | :-----------------------: |
|  llama-13b   |       92749.82(old)       |
| baichuan-13b |       79765.50(old)       |
|   qwen-14b   |       80749.57(old)       |
|   qwen-moe   |             -             |
|  internlm2   |             -             |
|  internvl2   |             -             |

***Note**: The training speed of each model was measured on 64 NVIDIA A100-PCIE-40GB GPUs linked by 100Gb/s bandwidth of InfiniBand with data type of bfloat16 and batch token size of 2048\*2048 (batch_size\*sequence_length,  batch_size = micro_batch_size \* gradient_accumulation_steps).*

|  Model   | Training Speed (tokens/s) |
| :------: | :-----------------------: |
| llama-7b |         26335.232         |

***Note**: Measured on 8 NVIDIA A100-PCIE-40GB GPUs with data type of bfloat16 and batch token size of 2304\*2048.*

## Inference

vLLm is quoted here for Inference.

### Batch Inference

```shell
python batch_infer.py \
    --model Qwen1.5-14B-Chat-Finetune \
    --prompt_file prompt.txt
```

### API Server

Start the server:

```shell
python server.py --model Qwen1.5-14B-Chat-Finetune
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
