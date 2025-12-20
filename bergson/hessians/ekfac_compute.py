import gc
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.data import IndexConfig, create_index, load_gradients, pad_and_tensor
from bergson.hessians.collector import (
    CovarianceCollector,
    HookCollectorBase,
    LambdaCollector,
)
from bergson.hessians.logger import get_logger
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils import get_device


class EkfacComputer:
    """Compute all factors for the EKFAC algorithm as described in https://arxiv.org/abs/2308.03296 Section 2.2.
    For all MLP modules in the model we do the following
    (where we use the notation torch.nn.Linear.weight.shape = [out, in]):

    1. Compute the covariance of the activations A_l (shape [in,in])
    and pseudo-gradients S_{l-1} (shape [out,out]) (Eq. 16).

    2. Compute the eigendecomposition of the covariance matrices Q_A (shape [in,in])
    and Q_S (shape [out,out]) (Eq. 18).

    3. Compute the eigenvalue correction Lambda (shape [out,in]) (Eq.20).

    If using FSDP, we will do everything in a fully sharded manner to save memory.
    As a result, we will save and load shards of tensors to reduce memory usage.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        data: Dataset,
        *,
        batches: list[list[int]],
        target_modules: set[str] | None = None,
        cfg: IndexConfig,
    ):
        self.model = model

        self.device = model.device
        self.dtype = model.dtype
        self.target_info = HookCollectorBase.discover_targets(model.base_model, target_modules)  # type: ignore

        self.target_modules = target_modules  # which modules to compute EKFAC for, by default uses all MLPs
        self.data = data
        self.batches = batches
        self.path = os.path.join(cfg.run_path, "influence_results")
        os.makedirs(self.path, exist_ok=True)

        self.cfg = cfg

        self.logger = get_logger(
            "EkfacComputer", level="DEBUG" if cfg.debug else "INFO"
        )

        ### Distributed related
        self.shard_computer = ShardedMul(
            target_info=self.target_info, lambda_damp_factor=cfg.lambda_damp_factor
        )
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.logger.info(
            f"Computing EKFAC for {list(self.target_info)} target modules."
        )

    def compute_covariance(self):
        cov_collector = CovarianceCollector(
            self.model.base_model,
            target_modules=self.target_modules,
            dtype=self.dtype,
            shard_computer=self.shard_computer,
            rank=self.rank,
            path=self.path,
        )

        self._collector(cov_collector, desc="covariances")

    def compute_eigendecomposition(
        self, covariance_type: Literal["activation", "gradient"]
    ):
        """This is Eq. 18 from above reference."""
        total_processed = torch.load(
            os.path.join(self.path, "total_processed_covariances.pt"),
            map_location=torch.device(get_device(self.rank)),
        )

        random.seed(0)
        shuffled_target_info = random.sample(
            list(self.target_info), len(list(self.target_info))
        )

        target_info_rank = shuffled_target_info[self.rank :: self.world_size]

        covariance_eigenvectors = {}

        for key in tqdm(
            target_info_rank,
            disable=False,
            desc=f"Rank {self.rank}: Computing {covariance_type} eigenvectors",
            position=self.rank,
            leave=False,
        ):
            matrix = self.shard_computer._compute_full_matrix(
                key,
                shard_path=os.path.join(
                    self.path, f"{covariance_type}_covariance_sharded"
                ),
            )  # type: ignore

            original_dtype = matrix.dtype
            matrix_normalized = matrix.to(torch.float64) / total_processed
            matrix_normalized = (matrix_normalized + matrix_normalized.T).div(2)

            # check if matrix_normalized is has NaNs or Infs
            if not torch.isfinite(matrix_normalized).all():
                raise ValueError(
                    f"Covariance matrix for {key} of type {covariance_type} contains NaNs or Infs."
                    "Consider increasing to fp32."
                )
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)

            except Exception as e:
                raise RuntimeError(
                    f"Eigendecomposition failed for {key} of type {covariance_type}"
                ) from e

            eigenvectors = eigenvectors.to(original_dtype).to(device="cpu").contiguous()
            covariance_eigenvectors[key] = eigenvectors

        covariance_eigenvectors = self.shard_computer._merge_and_shard_dict(
            input_dict=covariance_eigenvectors,
            covariance_type=covariance_type,
            dtype=self.dtype,
        )

        eigen_path = os.path.join(self.path, f"{covariance_type}_eigen_sharded")

        os.makedirs(eigen_path, exist_ok=True)

        save_file(
            covariance_eigenvectors,
            os.path.join(eigen_path, f"shard_{self.rank}.safetensors"),
        )

        self.logger.info(f"Saved {covariance_type} eigenvectors to {eigen_path}")
        gc.collect()
        torch.cuda.empty_cache()

    def compute_eigenvalue_correction(
        self,
    ):
        """
        This is Eq. 20 from above reference.
        """

        ev_correction_collector = LambdaCollector(
            model=self.model.base_model,
            target_modules=self.target_modules,
            shard_computer=self.shard_computer,
            rank=self.rank,
            path=self.path,
            world_size=self.world_size,
            device=self.device,
        )
        self._collector(ev_correction_collector, desc="lambda_correction")

    def _setup_profiler(self):
        """Set up profiler if profiling is enabled."""
        if not self.cfg.profile:
            return nullcontext()

        trace_handler = tensorboard_trace_handler(
            dir_name="profiler_logs", worker_name=f"rank_{self.rank}", use_gzip=True
        )
        my_schedule = schedule(wait=0, warmup=0, active=4, repeat=1)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=trace_handler,
            schedule=my_schedule,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,
        )

        log_dir = "profiler_logs"
        os.makedirs(log_dir, exist_ok=True)

        return prof

    def _collector(self, collector, desc: Optional[str] = None):
        total_processed = torch.tensor(0, device=self.model.device)
        prof = self._setup_profiler()
        step = 0

        with prof:
            for sl in tqdm(
                self.batches, disable=self.rank != 0, desc=f"Computing {desc}"
            ):
                batch = self.data[sl]
                x, y, valid_masks = pad_and_tensor(
                    batch["input_ids"],  # type: ignore
                    labels=batch.get("labels"),  # type: ignore
                    device=self.model.device,
                )

                total_processed += valid_masks.sum()
                collector.set_valid_masks(valid_masks)

                with (
                    collector,
                    (
                        record_function(f"step_{step}")
                        if self.cfg.profile
                        else nullcontext()
                    ),
                ):
                    logits = self.model(x).logits
                    logits = logits[:, :-1].reshape(-1, logits.size(-1))

                    if not self.cfg.sample:
                        losses = F.cross_entropy(
                            logits,
                            y[:, 1:].flatten(),
                            reduction="none",
                        ).reshape_as(y[:, 1:])
                    else:
                        with torch.no_grad():
                            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                            sampled_labels = torch.multinomial(
                                probs,
                                num_samples=1,
                            ).flatten()
                            del probs

                        flat_mask = valid_masks[:, :-1].flatten()
                        losses = F.cross_entropy(
                            logits[flat_mask],
                            sampled_labels[flat_mask],
                            reduction="none",
                        )

                    losses.sum().backward()
                    self.model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                if self.cfg.profile:
                    assert isinstance(prof, profile), "Profiler is not set up correctly"
                    prof.step()
                step += 1

        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            torch.save(
                total_processed, os.path.join(self.path, f"total_processed_{desc}.pt")
            )
        self.logger.info(f"Total processed: {total_processed.item()}")


class EkfacApplicator:
    def __init__(
        self,
        cfg: IndexConfig,
    ):
        self.cfg = cfg
        self.path = os.path.join(cfg.ekfac_path, "influence_results")
        self.gradient_path = cfg.gradient_path

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )

        ### Distributed related
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = f"cuda:{self.rank}"

        self.sharded_computer = ShardedMul(
            target_info=None, lambda_damp_factor=cfg.lambda_damp_factor
        )

        match cfg.precision:
            case "bf16":
                self.dtype = torch.bfloat16
            case "fp16":
                self.dtype = torch.float16
            case "fp32":
                self.dtype = torch.float32
            case "int4" | "int8":
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            case other:
                raise ValueError(f"Unsupported precision: {other}")

    def prepare_attribution(self):
        self.logger.info("Preparing EKFAC factors for attribution...")
        eigen_a = load_file(
            self.path + f"/activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        random_eigen_a = {}
        random_eigen_g = {}

        p = self.cfg.projection_dim

        for name in eigen_a.keys():
            proj_pi = self._projection(
                name=name,
                m=p,  # type: ignore
                n=eigen_a[name].shape[1],
                side="right",
                dtype=eigen_a[name].dtype,
            )

            proj_shards_wpt = torch.chunk(
                proj_pi, self.world_size, dim=1
            )  # (w, p, i/w)
            result_shard_pi = torch.einsum(
                "t i, p t-> p i", eigen_a[name], proj_shards_wpt[self.rank]
            ).contiguous()

            (
                dist.all_reduce(result_shard_pi, op=dist.ReduceOp.SUM)
                if dist.is_initialized()
                else None
            )

            shard_size = result_shard_pi.shape[0] // self.world_size
            start_row = self.rank * shard_size
            end_row = (self.rank + 1) * shard_size

            random_eigen_a[name] = result_shard_pi[start_row:end_row, :]

        random_activation_path = os.path.join(
            self.path, "random_activation_eigen_sharded"
        )
        os.makedirs(random_activation_path, exist_ok=True)
        save_file(
            random_eigen_a,
            os.path.join(random_activation_path, f"shard_{self.rank}.safetensors"),
        )

        for name in eigen_g.keys():
            proj_qo = self._projection(
                name=name,
                m=p,  # type: ignore
                n=eigen_g[name].shape[1],
                side="left",
                dtype=eigen_g[name].dtype,
            )
            proj_shards_wqr = torch.chunk(
                proj_qo, self.world_size, dim=1
            )  # (w, q, o/w)
            result_shard_qo = torch.einsum(
                "q r, r o -> q o", proj_shards_wqr[self.rank], eigen_g[name]
            ).contiguous()
            (
                dist.all_reduce(result_shard_qo, op=dist.ReduceOp.SUM)
                if dist.is_initialized()
                else None
            )

            shard_size = result_shard_qo.shape[0] // self.world_size
            start_row = self.rank * shard_size
            end_row = (self.rank + 1) * shard_size
            random_eigen_g[name] = result_shard_qo[start_row:end_row, :]

        random_gradient_path = os.path.join(self.path, "random_gradient_eigen_sharded")
        os.makedirs(random_gradient_path, exist_ok=True)
        save_file(
            random_eigen_g,
            os.path.join(random_gradient_path, f"shard_{self.rank}.safetensors"),
        )

        self.logger.info(
            f"Saved random activation eigenvectors to {random_activation_path}"
        )
        self.logger.info(
            f"Saved random gradient eigenvectors to {random_gradient_path}"
        )

        self.logger.info("-*-" * 50)

    def compute_ivhp_sharded(self):
        eigen_a = load_file(
            self.path + f"/activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        random_eigen_a = load_file(
            self.path
            + f"/random_activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        random_eigen_g = load_file(
            self.path + f"/random_gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        lambda_factor = load_file(
            self.path + f"/eigenvalue_correction_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        for k, v in lambda_factor.items():
            eigen_a[k] = eigen_a[k].to(dtype=torch.float32)
            eigen_g[k] = eigen_g[k].to(dtype=torch.float32)
            random_eigen_a[k] = random_eigen_a[k].to(dtype=torch.float32)
            random_eigen_g[k] = random_eigen_g[k].to(dtype=torch.float32)
            lambda_factor[k] = v.to(dtype=torch.float32)

        grad_sizes = {
            name: random_eigen_g[name].shape[0]
            * self.world_size
            * random_eigen_a[name].shape[0]
            * self.world_size
            for name in random_eigen_a
        }

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        # Allocate structured space ahead of time for the gradients
        grad_buffer = create_index(
            self.cfg.run_path,
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and computing IVHP..."
        )

        for i in tqdm(
            range(math.ceil(info["num_grads"] / self.cfg.gradient_batch_size)),
            desc="IVHP batches",
            disable=self.rank != 0,
        ):
            batch_slice = slice(
                i * self.cfg.gradient_batch_size,
                min((i + 1) * self.cfg.gradient_batch_size, info["num_grads"]),
            )
            # profile
            profiler = self._setup_profiler()
            with profiler as prof:
                transformed_gradients_slice = self.compute_ivhp_batch(
                    eigen_a=eigen_a,
                    mmap=mmap,
                    eigen_g=eigen_g,
                    lambda_factor=lambda_factor,
                    random_eigen_a=random_eigen_a,
                    random_eigen_g=random_eigen_g,
                    batch_slice=batch_slice,
                )
                torch.cuda.synchronize()
                for k, v in transformed_gradients_slice.items():
                    v = v.to(device="cpu", non_blocking=True).flatten(1).numpy()
                    grad_buffer[k][batch_slice] = v
                transformed_gradients_slice.clear()
                if prof is not None:
                    prof.step()

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")

    def compute_ivhp_batch(
        self,
        eigen_a,
        mmap,
        eigen_g,
        lambda_factor,
        random_eigen_a,
        random_eigen_g,
        batch_slice,
    ):
        transformed_gradients: dict[str, Tensor] = {}
        for k, v in eigen_a.items():
            gradients_noi = torch.from_numpy(mmap[k][batch_slice]).to(
                device=self.device, dtype=torch.float32
            )  # shape [num_grads, out*in]
            gradients_noi = gradients_noi.view(
                -1, eigen_g[k].shape[1], eigen_a[k].shape[1]
            )
            transformed_gradients[k] = self.sharded_computer._matmul(
                vector_nsa=gradients_noi, matrix_cb=v
            )

        self.logger.debug("Finished G @ Q_A")

        del eigen_a
        gc.collect()
        torch.cuda.empty_cache()

        for k, v in eigen_g.items():
            transformed_gradients[k] = self.sharded_computer._matmul(
                vector_nsa=transformed_gradients[k].transpose(-2, -1), matrix_cb=v
            ).transpose(-2, -1)

        self.logger.debug("Finished G'=Q_S.T @ G @ Q_A")
        del eigen_g
        gc.collect()
        torch.cuda.empty_cache()

        for k, v in lambda_factor.items():
            self.sharded_computer._hadamard(
                matrix_noi=transformed_gradients[k], lambda_ci=v
            )  # this is in-place

        self.logger.debug("Finished G'/lambda")

        for k, v in random_eigen_a.items():
            transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed_gradients[k],
                matrix_cb=v,
            )

        self.logger.debug("Finished G'/lambda @ P_A.T")

        for k, v in random_eigen_g.items():
            transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed_gradients[k].transpose(-2, -1),
                matrix_cb=v,
            ).transpose(-2, -1)

        self.logger.debug("Finished P_S.T @ G'/lambda @ P_A.T")
        return transformed_gradients

    def _projection(
        self,
        name: str,
        m: int,
        n: int,
        side: Literal["left", "right"],
        dtype: torch.dtype,
        projection_type: Literal["normal", "rademacher"] = "rademacher",
    ) -> Tensor:
        """Create a projection matrix deterministically based on identifier and side."""
        # Seed the PRNG with the name of the layer and what "side" we are projecting
        device = self.device
        if m == 0:
            A = torch.eye(n, dtype=dtype, device=device)

        message = bytes(f"{name}/{side}", "utf-8")
        digest = hashlib.md5(message).digest()
        seed = int.from_bytes(digest, byteorder="big") % (2**63 - 1)

        if projection_type == "normal":
            prng = torch.Generator(device).manual_seed(seed)
            A = torch.randn(m, n, device=device, dtype=dtype, generator=prng)
        elif projection_type == "rademacher":
            numpy_rng = np.random.Generator(np.random.PCG64(seed))
            random_bytes = numpy_rng.bytes((m * n + 7) // 8)
            random_bytes = np.frombuffer(random_bytes, dtype=np.uint8)
            A = np.unpackbits(random_bytes)[: m * n].reshape((m, n))
            A = torch.from_numpy(A).to(device, dtype=dtype)
            A = A.add_(-0.5).mul_(2)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        A /= A.norm(dim=1, keepdim=True)
        return A

    def _setup_profiler(self):
        """Set up profiler if profiling is enabled."""
        if not self.cfg.profile:
            return nullcontext()

        trace_handler = tensorboard_trace_handler(
            dir_name="profiler_logs", worker_name=f"rank_{self.rank}", use_gzip=True
        )
        my_schedule = schedule(wait=0, warmup=0, active=4, repeat=1)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=trace_handler,
            schedule=my_schedule,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,
        )

        log_dir = "profiler_logs"
        os.makedirs(log_dir, exist_ok=True)
        self.logger.info(f"Profiler logs will be saved to {log_dir}")

        return prof
