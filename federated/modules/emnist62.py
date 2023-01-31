import jax
import jax.numpy as jnp
import haiku as hk
import fedjax
from collections import defaultdict
from typing import List, Dict, Any, Optional

from . import base
from . import tree_structures
NUM_CLASSES = 62


class ConvDropout(base.Module):

    @staticmethod
    def get_dim() -> int:
        return tree_structures.ConvDropoutTreeStructure[-1]["i_1"]

    @staticmethod
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        model = fedjax.models.emnist.ConvDropoutModule(
            num_classes=NUM_CLASSES)
        return model(Xs, is_train=is_training)

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        packed_params, _ = base.pack_tree(unpacked_params)
        return packed_params

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        unpacked_params = base.unpack_tree(
            packed_params,
            tree_structures.ConvDropoutTreeStructure)
        return unpacked_params
