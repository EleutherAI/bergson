"""Stream output-token scores without a token-by-parameter gradient index."""

import json
import os
import heapq
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from bergson.collector.collector import HookCollectorBase
from bergson.data import _allocate_batches_world, pad_and_tensor
from bergson.score.output_influence import output_token_influence, query_directions
from bergson.score.output_model import install_output_loss_head, register_output_attention
from bergson.score.score_writer import MemmapSequenceScoreWriter, MemmapTokenScoreWriter
from bergson.utils.worker_utils import setup_model_and_peft


def output_influence_worker(rank, local_rank, world_size, index_cfg, score_cfg, preprocess_cfg, ds):
    from bergson.score.score import get_query_grads

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size,
                                device_id=torch.device('cuda', 0), timeout=timedelta(hours=2))
    register_output_attention()
    # This scorer runs FP32, as explicitly requested. Original FP64 factor
    # files, if any, remain unchanged; query conversion is declared here.
    query, _ = get_query_grads(score_cfg)
    query = {name: value.to(torch.float32) for name, value in query.items() if name in score_cfg.modules}
    columns = next(iter(query.values())).shape[0]
    if not score_cfg.offload_queries:
        query = {name: value.to('cuda:0') for name, value in query.items()}
    cfg = index_cfg
    cfg.model_kwargs = 'attn_implementation=bergson_output_jvp,use_cache=False'
    cfg.checkpoint_mlp = False
    cfg.checkpoint_attention = False
    model, targets = setup_model_and_peft(cfg)
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FP32 model parameters required for output scoring')
    install_output_loss_head(model)
    target_info = HookCollectorBase.discover_targets(model.base_model, targets, cfg.include_bias, cfg.filter_modules)
    directions = query_directions(model, {name: value.T for name, value in query.items()}, target_info)
    device = next(model.parameters()).device
    path = cfg.partial_run_path
    writer = (MemmapTokenScoreWriter(path, ds, columns, dtype=torch.float32)
              if cfg.attribute_tokens else MemmapSequenceScoreWriter(path, len(ds), columns, dtype=torch.float32))
    flags_path = path/'output_written.npy'
    if rank == 0 and not flags_path.exists():
        flags = np.lib.format.open_memmap(flags_path, mode='w+', dtype=np.bool_, shape=(len(ds),))
        flags[:] = False
        flags.flush()
    if dist.is_initialized():
        dist.barrier()
    flags = np.load(flags_path, mmap_mode='r+')
    # Output workers are independent replicas, so they may take unequal steps.
    # Build one exact global allocation, then assign each batch once; the FSDP
    # allocator's world-size divisibility constraint is not needed here.
    lengths = ds['length'][:]
    if cfg.loss_fn == 'dpo':
        n = len(lengths)//2
        pair_lengths = [max(lengths[i],lengths[n+i]) for i in range(n)]
        pair_batches = _allocate_batches_world(
            pair_lengths,cfg.token_batch_size//2,1,ranks=[0],
            max_batch_size=cfg.max_batch_size//2 if cfg.max_batch_size else None,
        )[0]
        batches = [indices+[n+i for i in indices] for indices in pair_batches]
    else:
        batches = _allocate_batches_world(lengths,cfg.token_batch_size,1,ranks=[0],
                                          max_batch_size=cfg.max_batch_size)[0]
    # Balance estimated linear/attention work without requiring equal step counts.
    # The coefficient matches the measured short-vs-16k cost scale; it changes
    # scheduling only, never examples, weights, or token losses.
    costs=[sum(lengths[i]**2+32768*lengths[i] for i in indices) for indices in batches]
    allocation=[[] for _ in range(world_size)];heap=[(0,r) for r in range(world_size)]
    for index in sorted(range(len(batches)),key=lambda i:(-costs[i],i)):
        load,owner=heapq.heappop(heap)
        allocation[owner].append(batches[index]);heapq.heappush(heap,(load+costs[index],owner))
    batches=allocation[rank]
    random.Random(42+rank).shuffle(batches)
    for indices in tqdm(batches,desc='Scoring output token influence'):
        if flags[indices].all():
            continue
        batch = ds[indices]
        x, y, _ = pad_and_tensor(batch['input_ids'], labels=batch['labels'], device=device, sync_max_len=False)
        weights = batch.get(cfg.example_weight_column) if cfg.example_weight_column else batch.get('advantage')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=score_cfg.output_compute_precision == 'bf16', cache_enabled=False):
            rates, _ = output_token_influence(model, x, y, directions, cfg, weights,
                                              reference_logps=batch.get('reference_logps'), sequence_lengths=batch['length'])
        if not torch.isfinite(rates).all():
            raise ValueError('Nonfinite output-token influence')
        values = rates[(y[:, 1:] != -100).to(rates.device)] if cfg.attribute_tokens else rates.sum(dim=1)
        writer(indices, values)
        writer.flush()
        flags[indices] = True
        flags.flush()
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        if not flags.all():
            raise ValueError('Output scores contain unwritten rows')
        info_path = path/'info.json'
        info = json.loads(info_path.read_text())
        info.update(token_influence='output', scoring_precision='float32', compute_precision=score_cfg.output_compute_precision,
                    position_convention='row t scores label t+1; masked labels excluded',
                    dpo_sides_separate=cfg.loss_fn == 'dpo', written_rows=len(ds))
        info_path.write_text(json.dumps(info, indent=2)+'\n')
        ds.save_to_disk(str(path/'data.hf'))
