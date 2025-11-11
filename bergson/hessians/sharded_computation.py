import gc
import os
from typing import Literal

import torch
import torch.distributed as dist
from jaxtyping import Float
from safetensors import safe_open
from torch import Tensor

from bergson.utils import get_device


class ShardedMul:
    def __init__(self, target_info, lambda_damp_factor=0.1):
        self.dist = dist.is_initialized()

        self.rank = dist.get_rank() if self.dist else 0
        self.world_size = dist.get_world_size() if self.dist else 1
        self.device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )

        self.target_info = target_info
        self.lambda_damp_factor = lambda_damp_factor

    def _init_covariance_dict(
        self,
        activation_covariance_dict: dict,
        gradient_covariance_dict: dict,
        dtype: torch.dtype,
    ):
        """This function initializes the covariance matrices for activations and gradients."""

        for name, (device, weight_shape) in self.target_info.items():
            # Activation covariance A^T A has shape [in_dim, in_dim]
            in_dim = weight_shape[1]
            shard_in_dim = in_dim if not self.dist else in_dim // self.world_size
            activation_covariance_dict[name] = torch.zeros(
                (shard_in_dim, in_dim), device=self.device, dtype=dtype
            )

            # Gradient covariance G^T G has shape [out_dim, out_dim]
            out_dim = weight_shape[0]
            shard_out_dim = out_dim if not self.dist else out_dim // self.world_size
            gradient_covariance_dict[name] = torch.zeros(
                (shard_out_dim, out_dim), device=self.device, dtype=dtype
            )

    def _matmul(
        self,
        vector_nsa: Float[Tensor, "n s a"],
        matrix_cb: Float[Tensor, "c b"],
    ) -> Float[Tensor, "n s b"]:
        """Vector-matrix multiplication.
        - If not distributed, this does usual multiplication with a=c.
        - If distributed, assumes that c=a/world_size and does sharded multiplication.
        """

        assert (
            vector_nsa.shape[2] == matrix_cb.shape[0] * self.world_size
        ), f"Vector shape {vector_nsa.shape} not compatible with matrix shape {matrix_cb.shape} and world_size {self.world_size}"

        if not self.dist:
            result_nsb = torch.einsum("n s c, c b-> n s b", vector_nsa, matrix_cb)

        else:
            result_nsb = self._sharded_matmul(vector_nsa, matrix_cb)

        return result_nsb

    def _transpose_matmul(
        self,
        vector_nsa: Float[Tensor, "n s a"],
        matrix_cb: Float[Tensor, "c b"],
    ) -> Float[Tensor, "n s b"]:
        if not self.dist:
            result_nsb = torch.einsum("n s c, b c -> n s b", vector_nsa, matrix_cb)
        else:
            result_nsb = self._sharded_transpose_matmul(vector_nsa, matrix_cb)
        return result_nsb

    def _compute_full_matrix(
        self,
        name: str,
        shard_path: str | os.PathLike,
    ) -> Tensor:
        """
        Load a full matrix from sharded covariance files. Needed to compute eigendecomposition.
        """

        files = os.listdir(shard_path)
        assert (
            len(files) == self.world_size
        ), f"Expected {self.world_size} shards, found {len(files)} in {shard_path}"

        device = get_device(self.rank)
        full_matrix = None

        if not self.dist:
            full_path_rank = os.path.join(
                shard_path, "shard_0.safetensors"
            )  # TODO: Does this work with different CUDA visible devices?
            with safe_open(full_path_rank, framework="pt", device=device) as f:
                full_matrix = f.get_tensor(name)

        else:
            full_matrix_list = []
            for shard_id in range(self.world_size):
                shard_path_rank = os.path.join(
                    shard_path, f"shard_{shard_id}.safetensors"
                )
                with safe_open(shard_path_rank, framework="pt", device=device) as f:
                    local_matrix = f.get_tensor(name)

                full_matrix_list.append(local_matrix)

            # Concatenate all shards to form the full matrix
            full_matrix = torch.cat(full_matrix_list, dim=0)

        return full_matrix

    def _merge_and_shard_dict(
        self,
        input_dict: dict[str, torch.Tensor],
        covariance_type: Literal["activation", "gradient"],
        dtype,
    ) -> dict[str, torch.Tensor]:
        """This function takes a dict of tensors, where each rank will have *full* eigenvectors of *some* modules.
        It then redistributes the tensors across all ranks,
        so that each rank has a *shard* of the eigenvectors of *each* module.
        """
        result_dict = {}
        if not self.dist:
            result_dict = input_dict
        else:
            for key in self.target_info:
                d_out, d_in = self.target_info[key][1]
                d = d_in if covariance_type == "activation" else d_out

                if key not in input_dict:
                    tensor = torch.zeros([d, d], device=self.device, dtype=dtype)
                else:
                    tensor = input_dict[key].to(device=self.device)

                shard_size = d // self.world_size
                (
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    if dist.is_initialized()
                    else None
                )

                shard = torch.empty(shard_size, d, device=self.device, dtype=dtype)
                shard.copy_(
                    tensor[self.rank * shard_size : (self.rank + 1) * shard_size, :]
                )
                result_dict[key] = shard.to(device="cpu", non_blocking=True)

                assert (
                    shard.shape[0] == shard_size
                ), f"Shard shape {shard.shape} does not match expected {shard_size}"

                del tensor

                gc.collect()
                torch.cuda.empty_cache()

        return result_dict

    def _hadamard(
        self, matrix_noi: Float[Tensor, "n o i"], lambda_ci: Float[Tensor, "c i"]
    ):
        if not self.dist:
            global_lambda_mean = lambda_ci.mean()
            inverse_lambda = (
                lambda_ci + self.lambda_damp_factor * global_lambda_mean
            ).reciprocal()
            matrix_noi.mul_(inverse_lambda)
        else:
            self._sharded_hadamard(matrix_noi, lambda_ci)

    def _sharded_matmul(
        self,
        vector_nsa: Float[Tensor, "n s a"],
        matrix_cb: Float[Tensor, "c b"],
    ) -> Float[Tensor, "n s b"]:
        """
        Sharded matrix multiplication for distributed training. Assumes that c= a/world_size.
        vector: [n, s, a]
        matrix_shard: [a/world_size, b]
        Returns: [n, s, b]
        """
        # Split the vector into shards
        vector_shards_wnsc = torch.chunk(
            vector_nsa, self.world_size, dim=-1
        )  # (w, n, s, a/w)
        n, s, b = vector_nsa.shape[0], vector_nsa.shape[1], matrix_cb.shape[1]

        result_nsb = torch.zeros(
            (n, s, b),
            device=vector_nsa.device,
            dtype=vector_nsa.dtype,
        )

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_cb = matrix_cb
            else:
                shard_cb = torch.zeros_like(matrix_cb)

            dist.broadcast(shard_cb, src=rank_index)
            result_nsb += torch.einsum(
                "n s c, c b-> n s b", vector_shards_wnsc[rank_index], shard_cb
            )  # [B, c]
            if self.rank != rank_index:
                del shard_cb

        return result_nsb

    def _sharded_hadamard(
        self, matrix_noi: Float[Tensor, "n o i"], lambda_ci: Float[Tensor, "c i"]
    ):
        """
        Sharded in-place element-wise multiplication for distributed training.
        gradients: [n, o, i]
        matrix_shard: [c, i] where c=o/world_size

        """

        global_lambda_mean = lambda_ci.mean()

        dist.all_reduce(global_lambda_mean, op=dist.ReduceOp.SUM)
        global_lambda_mean /= self.world_size

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_ci = lambda_ci
            else:
                shard_ci = torch.zeros_like(lambda_ci)

            dist.broadcast(shard_ci, src=rank_index)

            start_row = rank_index * shard_ci.shape[0]
            end_row = (rank_index + 1) * shard_ci.shape[0]
            inverse_lambda = (
                shard_ci + self.lambda_damp_factor * global_lambda_mean
            ).reciprocal()

            matrix_noi[:, start_row:end_row, :].mul_(inverse_lambda)

            if self.rank != rank_index:
                del shard_ci

    def _sharded_transpose_matmul(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        matrix_bc: Float[Tensor, "b c"],
    ):
        """
        Sharded matrix multiplication for distributed training.
        Assumes that c=i/world_size if left or o/world_size if right.
        gradients: [n, o, i]
        matrix_shard: [c, b] where b=i if left or b=o if right
        Returns: [n, o, c*w] if left or [n, c*w, i] if right
        """

        x, y = (matrix_noi.shape[1], matrix_bc.shape[0] * self.world_size)

        result_nxy = torch.zeros(
            matrix_noi.shape[0], x, y, device=matrix_noi.device, dtype=matrix_noi.dtype
        )

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_bc = matrix_bc
            else:
                shard_bc = torch.zeros_like(matrix_bc)
            dist.broadcast(shard_bc, src=rank_index)

            shard_size = shard_bc.shape[0]
            start_row = rank_index * shard_size
            end_row = (rank_index + 1) * shard_size

            result_nxy[:, :, start_row:end_row].copy_(
                torch.einsum("n o i, c i -> n o c", matrix_noi, shard_bc)
            )

            if self.rank != rank_index:
                del shard_bc

        return result_nxy
