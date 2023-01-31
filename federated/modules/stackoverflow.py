import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_federated as tff
from collections import defaultdict
from typing import List, Dict, Any, Optional

from . import base
from . import lstm
from . import tree_structures


forward_fn, loss_fn = lstm.create_lstm_model()


class LSTM(base.Module):

    @staticmethod
    def get_dim() -> int:
        return tree_structures.LSTMTreeStructure[-1]["i_1"]

    @staticmethod
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        return forward_fn({"x": Xs, "y": Ys})

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        packed_params, _ = base.pack_tree(unpacked_params)
        return packed_params

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        unpacked_params = base.unpack_tree(
            packed_params,
            tree_structures.LSTMTreeStructure)
        return unpacked_params

    @staticmethod
    def sequence_cross_entropy_loss(
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        num_classes: int,
    ) -> jnp.ndarray:
        return loss_fn({"y": labels}, preds=logits)


class LR(base.Module):

    @staticmethod
    def get_dim() -> int:
        return tree_structures.LRTreeStructure[-1]["i_1"]

    @staticmethod
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        lr_model = hk.nets.MLP([
            tff
            .simulation
            .baselines
            .stackoverflow
            .DEFAULT_TAG_VOCAB_SIZE])
        return lr_model(Xs)

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        packed_params, _ = base.pack_tree(unpacked_params)
        return packed_params

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        unpacked_params = base.unpack_tree(
            packed_params,
            tree_structures.LRTreeStructure)
        return unpacked_params
