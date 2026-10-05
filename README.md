# GPT-2 (124M) from scratch

https://huggingface.co/netrunner20/gpt2-124m-fineweb-edu

GPT-2 (124M) built and trained from scratch in PyTorch, following Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU). Trained on 10B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) on 8 × A100 GPUs.

Final validation loss was 3.07, lower than OpenAI's GPT-2 (124M) at 3.29 on the same validation data.

## Results

| Metric | Result |
|---|---|
| Validation loss | 3.073 |
| First below GPT-2 | step 6,750 of 19,073 (3.5B tokens) |
| Data | FineWeb-Edu sample-10BT: 9.95B tokens in 100 shards |
| Training | 19,073 steps × 524,288 tokens, 1 h 55 min on 8 × A100 80 GB |
| Speed | about 1.53M tokens/sec (343 ms per step) |

| Step | Tokens | Validation loss |
|---:|---:|---:|
| 250 | 0.13B | 6.1339 |
| 1,000 | 0.52B | 4.1614 |
| 2,000 | 1.05B | 3.6582 |
| 5,000 | 2.62B | 3.3563 |
| 6,750 | 3.54B | 3.2805 |
| 10,000 | 5.24B | 3.1973 |
| 15,000 | 7.86B | 3.1102 |
| 19,073 | 10.0B | 3.0731 |

### Samples

Prompt `Hello, I'm a language model,`, 32 tokens, top-50 sampling. Verbatim from [`logs/train_output.txt`](logs/train_output.txt):

| Step | Output |
|---:|---|
| 250 | Hello, I'm a language model, what I will be the I will need all we see with a problem but the way that what to get your child is |
| 1,000 | Hello, I'm a language model, we got a lot of excitement in education. We could take the best classes of teachers to use in classes of teachers to |
| 5,000 | Hello, I'm a language model, and all of these are just some of the data structures that I would like to look at. I like to ask all |
| 19,073 | Hello, I'm a language model, and I know that you should learn your langauge to a minimum. And I don't really want you to think |

## Speed-up implementations

Measured with single NVIDIA L4, micro batch 8 × 1,024 tokens, and each change added on top of the previous ones ([`bench.py`](bench.py)):

| Change | Tokens/sec | vs. baseline |
|---|---:|---:|
| Baseline (fp32) | 6,303 | 1.00× |
| TF32 matmuls | 8,754 | 1.39× |
| bf16 autocast | 10,957 | 1.74× |
| `torch.compile` | 23,688 | 3.76× |
| FlashAttention | 33,146 | 5.26× |
| Vocabulary 50,257 → 50,304 | 34,436 | 5.46× |
| AdamW betas/eps + gradient clipping | 33,956 | 5.39× |
| Learning-rate warmup + cosine decay | 33,914 | 5.38× |
| Weight decay + fused AdamW | 37,669 | 5.98× |
| Gradient accumulation (524,288 tokens per step) | 40,753 | 6.47× |

## Files

| File | Description |
|---|---|
| [`train_gpt2.py`](train_gpt2.py) | Model and training: DDP, gradient accumulation, warmup + cosine learning rate, validation, sampling, checkpoints with exact resume |
| [`fineweb.py`](fineweb.py) | Downloads FineWeb-Edu sample-10BT and writes 100 shards of 100M GPT-2 tokens |
| [`gpt-2.ipynb`](gpt-2-scalable.ipynb) | Colab notebook that writes the two scripts |
| [`logs/log.txt`](logs/log.txt) | Training and validation loss for every step |
| [`logs/train_output.txt`](logs/train_output.txt) | Full console output of the run, including samples every 250 steps |

## Reproduce

```bash
pip install torch tiktoken datasets tqdm numpy
python fineweb.py                                       # ~16 min, ~20 GB of token shards
torchrun --standalone --nproc_per_node=8 train_gpt2.py  # ~2 h on 8 × A100 80 GB
```

The defaults are the main run. On 40 GB GPUs add `--micro_batch 32`; on a single GPU run `python train_gpt2.py`.

## Credits

Built by following Andrej Karpathy's [video](https://www.youtube.com/watch?v=l8pRSuU81PU) and [build-nanogpt](https://github.com/karpathy/build-nanogpt). 
