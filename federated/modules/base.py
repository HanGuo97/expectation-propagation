import jax
import jax.numpy as jnp
import haiku as hk
from collections import defaultdict
from typing import Any, Tuple, List, Dict, Optional

class Module(object):

    @staticmethod
    def get_dim() -> int:
        raise NotImplementedError

    @classmethod
    def apply(
            cls,
            params: jnp.ndarray,
            rng: jax.random.KeyArray,
            Xs: jnp.ndarray,
            Ys: jnp.ndarray,
            is_training: Optional[bool] = None,  # not needed for now
    ) -> jnp.ndarray:
        unpacked_params = cls.unpack_params(params)
        return cls.forward.apply(
            unpacked_params,
            rng,
            Xs,
            Ys,
            is_training)

    @classmethod
    def init(
            cls,
            rng: jax.random.KeyArray,
            Xs: jnp.ndarray,
            Ys: jnp.ndarray,
            is_training: Optional[bool] = None,  # not needed for now
    ) -> jnp.ndarray:
        unpacked_params = cls.forward.init(
            rng,
            Xs,
            Ys,
            is_training)
        return cls.pack_params(unpacked_params)

    # https://github.com/google/jax/issues/1251
    # https://github.com/google/jax/issues/7702
    @staticmethod
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        raise NotImplementedError


def pack_tree(unpacked_tree: hk.Params) -> Tuple[jnp.ndarray, List[Dict[str, Any]]]:
    tree_structure = []
    packed_tree = jnp.zeros(shape=[0])
    for module_name, param_name, param in (
        hk.data_structures.traverse(unpacked_tree)
    ):
        i_0 = packed_tree.size
        packed_tree = jnp.concatenate(
            [packed_tree, param.flatten()],
            axis=0)
        i_1 = packed_tree.size
        tree_structure.append({
            "module_name": module_name,
            "param_name": param_name,
            "shape": param.shape,
            "size": param.size,
            "i_0": i_0,
            "i_1": i_1,
        })

    return packed_tree, tree_structure


def unpack_tree(
    packed_tree: jnp.ndarray,
    tree_structure: List[Dict[str, Any]],
) -> hk.Params:
    unpacked_tree: hk.Params = defaultdict(dict)
    for params_structure in tree_structure:
        module_name = params_structure["module_name"]
        param_name = params_structure["param_name"]
        shape = params_structure["shape"]
        i_0 = params_structure["i_0"]
        i_1 = params_structure["i_1"]
        unpacked_tree[module_name][param_name] = (
            packed_tree[i_0: i_1].reshape(shape)
        )

    return unpacked_tree
