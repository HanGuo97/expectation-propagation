import jax
import jax.numpy as jnp
import haiku as hk
from collections import defaultdict
from typing import List, Dict, Any, Optional

from . import base
from . import resnet
from . import tree_structures
NUM_CLASSES = 100


class ResNet(base.Module):

    @staticmethod
    def get_dim() -> int:
        return tree_structures.ResNetTreeStructure[-1]["i_1"]

    @staticmethod
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        model = resnet.ResNet18(
            num_classes=NUM_CLASSES,
            resnet_v2=True)
        return model(Xs, is_training=is_training)

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        packed_params, _ = base.pack_tree(unpacked_params)
        return packed_params

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        unpacked_params = base.unpack_tree(
            packed_params,
            tree_structures.ResNetTreeStructure)
        return unpacked_params


class SimpleModel(base.Module):

    @staticmethod
    def get_dim() -> int:
        return tree_structures.SimpleModelTreeStructure[-1]["i_1"]

    @staticmethod
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        lr_model = hk.nets.MLP([NUM_CLASSES])
        return lr_model(Xs)

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        packed_params, _ = base.pack_tree(unpacked_params)
        return packed_params

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        unpacked_params = base.unpack_tree(
            packed_params,
            tree_structures.SimpleModelTreeStructure)
        return unpacked_params
