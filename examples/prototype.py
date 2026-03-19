import gc
import os
import shutil
import tempfile

import torch
import torchopt
from datasets import load_dataset
from scipy.stats import spearmanr
from torchopt.pytree import tree_iter
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.data import tokenize_and_chunk
from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer, TrainerState
from bergson.utils.math import weighted_causal_lm_ce

# Disable autocast
torch.cuda.is_bf16_supported = lambda *a, **k: False


def make_gpt2_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    # untie weights, otherwise bergson blows up
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
    model.to(device)
    return model


def run_test(
    max_length, train_ds, test_ids, batch_size, device, num_subsets=20, seed=42
):
    n_train = (len(train_ds) // batch_size) * batch_size
    input_ids = torch.tensor([test_ids], device=device)

    # Save pretrained params/buffers for LDS retraining
    init_model = make_gpt2_model(device)
    init_params = {
        k: v.detach().clone()
        for k, v in init_model.named_parameters(remove_duplicate=False)
        if v.requires_grad
    }
    init_buffers = {
        k: v.detach().clone()
        for k, v in init_model.named_buffers(remove_duplicate=False)
    }
    del init_model

    # MAGIC forward pass
    model = make_gpt2_model(device)
    torch.manual_seed(seed)
    optimizer = torchopt.adamw(1e-5, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    ckpt_dir = tempfile.mkdtemp()

    train_stream = DataStream(
        train_ds,
        batch_size=batch_size,
        device=device,
    )
    fwd_state = trainer.train(fwd_state, train_stream, inplace=True, save_dir=ckpt_dir)
    fwd_state.save(os.path.join(ckpt_dir, "final_state.ckpt")).result()

    # MAGIC backward pass
    bwd_stream = DataStream(
        train_ds,
        batch_size=batch_size,
        device=device,
    )
    with fwd_state.activate(model) as params:
        test_loss = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        ).loss
        query_grads = grad_tree(test_loss, params)
        query_grads = {k: g.detach().clone() for k, g in query_grads.items()}
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(fwd_state.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        bwd_state = BackwardState(
            query_grads, opt_grads, torch.zeros_like(bwd_stream.weights)
        )

    bwd_state = trainer.backward(
        ckpt_dir, bwd_stream, bwd_state, fwd_state, inplace=True, cleanup=True
    )
    scores = bwd_state.weight_grads.detach().cpu()

    # Baseline: eval loss from the fully-trained forward pass
    with torch.no_grad(), fwd_state.activate(model):
        baseline_loss = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        ).loss.item()

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    del trainer, fwd_state, bwd_state, query_grads, opt_grads, test_loss
    gc.collect()
    torch.cuda.synchronize()

    # LDS: leave-subset-out retraining
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_train, generator=gen)
    subsets = perm.chunk(num_subsets)

    loss_diffs = []
    score_sums = []
    for i, subset in enumerate(subsets):
        torch.manual_seed(seed)

        params = {
            k: v.detach().clone().requires_grad_(False) for k, v in init_params.items()
        }
        opt = torchopt.adamw(1e-5, betas=(0.95, 0.975), eps_root=1e-2)
        subset_trainer = Trainer(model, opt)
        subset_state = TrainerState(
            params,
            opt.init(params),
            {k: v.detach().clone() for k, v in init_buffers.items()},
        )
        subset_stream = DataStream(
            train_ds,
            batch_size=batch_size,
            device=device,
        )
        subset_stream.weights.data.fill_(1.0)
        subset_stream.weights.data[subset] = 0.0

        for batch in subset_stream:
            subset_state = subset_trainer.step(subset_state, batch)

        with torch.no_grad(), subset_state.activate(model):
            subset_loss = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone(),
            ).loss.item()

        loss_diffs.append(baseline_loss - subset_loss)
        score_sums.append(scores[subset].sum().item())

        running_rho = (
            spearmanr(loss_diffs, score_sums).statistic
            if len(loss_diffs) > 2
            else float("nan")
        )
        print(
            f"    subset {i+1}/{len(subsets)}: diff={loss_diffs[-1]:.6f}"
            f"  score_sum={score_sums[-1]:.6f}  running_rho={running_rho:.4f}"
        )

        del subset_trainer, subset_state, subset_stream
        gc.collect()
        torch.cuda.synchronize()

    rho = spearmanr(loss_diffs, score_sums).statistic

    print(
        f"  {max_length:4d} tok:  LDS Spearman={rho:.4f}  "
        f"n_subsets={num_subsets}  n_train={n_train}"
    )

    del model, init_params, init_buffers
    gc.collect()
    torch.cuda.synchronize()
    return rho, loss_diffs, score_sums


def main():
    rank = 0

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    raw_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    n_train = 100
    batch_size = 8

    results = {}
    for max_length in [512]:
        raw_ds = raw_ds.filter(lambda x: len(x["text"].strip()) > 0)
        ds = tokenize_and_chunk(raw_ds, tokenizer, max_length)
        tokens = ds["input_ids"][:]
        ds = ds.select(range(n_train))

        train_ds = ds.select(range(len(ds) - 1))
        test_ids = tokens[n_train - 1].tolist()

        rho, diffs, score_sums = run_test(
            max_length, train_ds, test_ids, batch_size, device
        )
        results[max_length] = rho

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for ml, rho in results.items():
        print(f"  {ml:4d} tok:  LDS Spearman={rho:.4f}")


if __name__ == "__main__":
    main()
