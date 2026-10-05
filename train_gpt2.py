# train_gpt2.py: train GPT-2 (124M) from scratch on FineWeb-Edu.
#
# One GPU:                python train_gpt2.py
# Several GPUs (DDP):     torchrun --standalone --nproc_per_node=8 train_gpt2.py
# Quick test (Colab L4):  torchrun --standalone --nproc_per_node=1 train_gpt2.py --data_root edu_fineweb_test \
#                             --micro_batch 8 --total_batch 65536 --max_steps 40 --warmup_steps 10 ...
#
# The default settings are the main run from the video:
#   524,288 tokens per step x 19,073 steps = 10B tokens (one pass over FineWeb-Edu sample-10BT).
# Needs the token shards made by fineweb.py (folder set by --data_root).

import argparse, contextlib, math, os, time
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

# ---------------------------------------------------------------------------
# Settings (defaults = the main run)

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='edu_fineweb10B')       # folder with the token shards from fineweb.py
parser.add_argument('--log_dir', default='log')                    # log.txt and checkpoints go here
parser.add_argument('--micro_batch', type=int, default=64)         # B: rows per GPU per forward pass (64 fits in 80GB)
parser.add_argument('--seq_len', type=int, default=1024)           # T: tokens per row
parser.add_argument('--total_batch', type=int, default=524288)     # tokens per optimizer step (2**19, GPT-3 paper)
parser.add_argument('--max_steps', type=int, default=19073)        # 19,073 x 524,288 = 10B tokens
parser.add_argument('--warmup_steps', type=int, default=715)       # 375M warmup tokens (GPT-3 paper) / 524,288
parser.add_argument('--max_lr', type=float, default=6e-4)          # GPT-3 Small learning rate
parser.add_argument('--min_lr_ratio', type=float, default=0.1)     # cosine decay down to 10% of max_lr
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--eval_interval', type=int, default=250)      # validation loss every N steps
parser.add_argument('--val_steps', type=int, default=20)           # batches per validation loss estimate
parser.add_argument('--sample_interval', type=int, default=250)    # print a few generated texts every N steps (0 = never)
parser.add_argument('--checkpoint_interval', type=int, default=5000)  # save model + optimizer every N steps (and at the end)
parser.add_argument('--compile', type=int, default=1)              # 1 = torch.compile the training model
parser.add_argument('--resume', default='')                        # path to a checkpoint to continue from
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# model

@dataclass
class GPTConfig:
    block_size: int = 1024   # context length
    vocab_size: int = 50257  # 256 bytes + 50,000 merged pieces + 1 special token (we use 50304, see below)
    n_layer: int = 12        # number of Blocks
    n_head: int = 12         # attention heads per Block
    n_embd: int = 768        # vector size for one token

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # q, k, v all at once
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)      # mixes the heads back together
        self.c_proj.NANOGPT_SCALE_INIT = 1                         # residual layer: start smaller (see _init_weights)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # FlashAttention
        y = y.transpose(1, 2).contiguous().view(B, T, C)                 # heads back into one 768 vector
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight sharing between input embedding and output layer
        self.apply(self._init_weights)                     # GPT-2 style starting weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5   # 0.02 / sqrt(24) for the residual layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        # weight decay only on 2D tensors (matmul weights, embeddings), not on biases / LayerNorm
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        use_fused = device_type == 'cuda'  # fused AdamW: one GPU kernel for the whole update
        if verbose:
            print(f"decayed tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
            print(f"non-decayed tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
            print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

# ---------------------------------------------------------------------------
# Data: reads the token shards written by fineweb.py

def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)  # uint16 on disk -> int32 -> torch long
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    # Each GPU (process) reads its own slice: rank r takes the r-th block of B*T tokens,
    # then all ranks jump forward together by B*T*num_processes tokens.
    def __init__(self, B, T, process_rank, num_processes, split, data_root, verbose=True):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split} in {data_root}"
        if verbose:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets (next tokens)
        self.current_position += B * T * self.num_processes
        # move to the next shard if the next batch would run past the end of this one
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def state(self):
        # where we are in the data (rank 0's view), saved in checkpoints so a resumed run continues here
        return {'shard': self.current_shard, 'position': self.current_position - self.B * self.T * self.process_rank}

    def load_state(self, state):
        self.current_shard = state['shard']
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = state['position'] + self.B * self.T * self.process_rank

# ---------------------------------------------------------------------------
# Setup: one process per GPU when launched with torchrun, otherwise a single process

ddp = int(os.environ.get('RANK', -1)) != -1  # torchrun sets RANK
if ddp:
    assert torch.cuda.is_available(), "DDP here needs CUDA"
    ddp_rank = int(os.environ['RANK'])               # this process's number among all processes
    ddp_local_rank = int(os.environ['LOCAL_RANK'])   # this process's GPU number on this machine
    ddp_world_size = int(os.environ['WORLD_SIZE'])   # total number of processes (= GPUs)
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    init_process_group(backend='nccl')
    master_process = ddp_rank == 0                   # only process 0 prints, logs and saves
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if device.startswith('cuda') else 'cpu'
if master_process:
    print(f"using device: {device}, world size: {ddp_world_size}")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision('high')  # TF32

B, T = args.micro_batch, args.seq_len
assert args.total_batch % (B * T * ddp_world_size) == 0, "total_batch must be divisible by B * T * number of GPUs"
grad_accum_steps = args.total_batch // (B * T * ddp_world_size)
if master_process:
    print(f"total batch: {args.total_batch:,} tokens | micro batch: {B} x {T} | GPUs: {ddp_world_size} "
          f"| gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'train', args.data_root, verbose=master_process)
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'val', args.data_root, verbose=master_process)

model = GPT(GPTConfig(vocab_size=50304))  # 50304 = 128 x 393: a "nice" size for the GPU
model.to(device)
raw_model = model  # the plain model: used for evaluation, text sampling and saving
if args.compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, device_type, verbose=master_process)

def get_lr(it):
    # linear warmup, then cosine decay to min_lr (GPT-3 paper)
    min_lr = args.max_lr * args.min_lr_ratio
    if it < args.warmup_steps:
        return args.max_lr * (it + 1) / args.warmup_steps
    if it > args.max_steps:
        return min_lr
    decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.max_lr - min_lr)

start_step = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    train_loader.load_state(ckpt['train_loader'])
    start_step = ckpt['step']
    if master_process:
        print(f"resumed from {args.resume}: continuing at step {start_step}")
    del ckpt

enc = tiktoken.get_encoding('gpt2')
os.makedirs(args.log_dir, exist_ok=True)
log_file = os.path.join(args.log_dir, 'log.txt')
if master_process and not args.resume:
    open(log_file, 'w').close()  # start a fresh log (a resumed run keeps adding to the old one)

def log(line):
    if master_process:
        with open(log_file, 'a') as f:
            f.write(line + '\n')

# ---------------------------------------------------------------------------
# Training loop
# The loop runs to step == max_steps: that last pass only evaluates and saves the finished model.

last_val_loss = None
t_train_start = time.time()
for step in range(start_step, args.max_steps + 1):
    last_step = (step == args.max_steps)

    # 1) validation loss
    if step % args.eval_interval == 0 or last_step:
        t_eval = time.time()
        raw_model.eval()
        val_loader.reset()
        val_loss_accum = torch.zeros((), device=device)
        with torch.no_grad():
            for _ in range(args.val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = raw_model(x, y)
                val_loss_accum += loss.detach() / args.val_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        last_val_loss = val_loss_accum.item()
        if master_process:
            print(f"step {step:5d} | val loss {last_val_loss:.4f} | took {time.time() - t_eval:.1f}s")
        log(f"{step} val {last_val_loss:.4f}")

    # 2) generate a few texts to see what the model writes
    if master_process and args.sample_interval > 0 and step > 0 and (step % args.sample_interval == 0 or last_step):
        raw_model.eval()
        num_return_sequences, max_length = 4, 32
        tokens = torch.tensor(enc.encode("Hello, I'm a language model,"), dtype=torch.long)
        xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = raw_model(xgen)
                logits = logits[:, -1, :].float()
                logits[:, 50257:] = -float('inf')  # the 47 padding tokens (50257..50303) are not real tokens
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # only sample from the 50 most likely
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            print(f"sample {i}: {enc.decode(xgen[i, :max_length].tolist())}")

    # 3) checkpoint: model + optimizer + data position, enough to resume training exactly
    if master_process and step > start_step and (step % args.checkpoint_interval == 0 or last_step):
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': asdict(raw_model.config),
            'step': step,
            'val_loss': last_val_loss,
            'train_loader': train_loader.state(),
        }
        checkpoint_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"saved {checkpoint_path}")

    if last_step:
        break

    # 4) one optimizer step = grad_accum_steps micro batches on every GPU
    raw_model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = torch.zeros((), device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with DDP, skip the gradient sync between GPUs until the last micro batch
        sync_context = model.no_sync() if (ddp and micro_step < grad_accum_steps - 1) else contextlib.nullcontext()
        with sync_context:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps  # each micro batch counts for 1/K of the full batch
            loss_accum += loss.detach()
            loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # average the printed loss over GPUs
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt
    if master_process:
        print(f"step {step:5d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f} "
              f"| dt {dt * 1000:.2f}ms | tok/sec {tokens_per_sec:,.0f}")
    log(f"{step} train {loss_accum.item():.6f}")

if master_process:
    print(f"done: {time.time() - t_train_start:.1f}s total")
if ddp:
    destroy_process_group()
