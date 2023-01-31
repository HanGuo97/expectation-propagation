import jax
import jax.numpy as jnp
import haiku as hk

from . import base

INPUT_SIZE = 24 * 24 * 3
LAYER_0_SIZE = 300
LAYER_1_SIZE = 100


class MLP(base.Module):

    @staticmethod
    def get_dim() -> int:
        return (
            (INPUT_SIZE + 1) * LAYER_0_SIZE +
            (LAYER_0_SIZE + 1) * LAYER_1_SIZE
        )

    @staticmethod
    @hk.without_apply_rng
    @hk.transform
    def forward(Xs: jnp.ndarray, Ys: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        model = hk.Sequential([
            hk.Linear(LAYER_0_SIZE, name="layer_0"), jax.nn.relu,
            hk.Linear(LAYER_1_SIZE, name="layer_1"),
        ])
        return model(Xs)

    @staticmethod
    def pack_params(unpacked_params: hk.Params) -> jnp.ndarray:
        return jnp.concatenate([
            unpacked_params["layer_0"]["w"].flatten(),
            unpacked_params["layer_0"]["b"],
            unpacked_params["layer_1"]["w"].flatten(),
            unpacked_params["layer_1"]["b"].flatten()], axis=0)

    @staticmethod
    def unpack_params(packed_params: jnp.ndarray) -> hk.Params:
        w0_index = INPUT_SIZE * LAYER_0_SIZE
        b0_index = w0_index + LAYER_0_SIZE
        w1_index = b0_index + LAYER_0_SIZE * LAYER_1_SIZE
        b1_index = w1_index + LAYER_0_SIZE
        return {
            "layer_0": {
                "w": packed_params[:w0_index].reshape(INPUT_SIZE, LAYER_0_SIZE),
                "b": packed_params[w0_index: b0_index].reshape(LAYER_0_SIZE),
            },
            "layer_1": {
                "w": packed_params[b0_index: w1_index].reshape(LAYER_0_SIZE, LAYER_1_SIZE),
                "b": packed_params[w1_index: b1_index].reshape(LAYER_1_SIZE),
            }
        }


def test_pack_unpack_params(params_0: jnp.ndarray) -> None:
    packed_params = params_0
    for _ in range(10):
        unpacked_params = MLP.unpack_params(packed_params)
        packed_params = MLP.pack_params(unpacked_params)

        print(
            (packed_params == params_0).all(),
            packed_params.shape == params_0.shape
        )

