import enum
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_federated as tff
from . import cifar100
from . import emnist62
from . import stackoverflow
from typing import Dict


class ModelIndex(enum.IntEnum):
    CIFAR100 = 0
    EMNIST62 = 1
    STACKOVERFLOW_LR = 2
    STACKOVERFLOW_NWP = 3
    CIFAR100_TOY = 4


class ReductionMode(enum.IntEnum):
    SUM = 0
    MEAN = 1


MODELS = [
    cifar100.ResNet,
    emnist62.ConvDropout,
    stackoverflow.LR,
    stackoverflow.LSTM,
    cifar100.SimpleModel,
]


def get_dim(
    index: ModelIndex,
) -> int:

    return MODELS[index].get_dim()


def init(
    index: ModelIndex,
    rng: jax.random.KeyArray,
    Xs: jnp.ndarray,
    Ys: jnp.ndarray,
) -> jnp.ndarray:

    return MODELS[index].init(
        rng=rng,
        Xs=Xs,
        Ys=Ys,
        is_training=True)


def apply_for_train(
    index: ModelIndex,
    params: jnp.ndarray,
    rng: jax.random.KeyArray,
    Xs: jnp.ndarray,
    Ys: jnp.ndarray,
) -> jnp.ndarray:

    return MODELS[index].apply(
        params=params,
        rng=rng,
        Xs=Xs,
        Ys=Ys,
        is_training=True)


def apply_for_eval(
    index: ModelIndex,
    params: jnp.ndarray,
    rng: jax.random.KeyArray,
    Xs: jnp.ndarray,
    Ys: jnp.ndarray,
) -> jnp.ndarray:

    return MODELS[index].apply(
        params=params,
        rng=rng,
        Xs=Xs,
        Ys=Ys,
        is_training=False)


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    one_hot_labels = jax.nn.one_hot(
        labels,
        num_classes=num_classes)
    return optax.softmax_cross_entropy(
        logits=logits,
        labels=one_hot_labels)


def binary_cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    losses = optax.sigmoid_binary_cross_entropy(
        logits=logits,
        labels=labels)
    # TFF uses "sum" loss, so we scale the loss
    # by the batch size to mimic the same behavior.
    losses = losses * losses.shape[0]
    return jnp.mean(losses, axis=-1)


def compute_loss(
    index: ModelIndex,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    if index == ModelIndex.STACKOVERFLOW_NWP:
        return stackoverflow.LSTM.sequence_cross_entropy_loss(
            logits=logits,
            labels=labels,
            num_classes=num_classes)
    elif index == ModelIndex.STACKOVERFLOW_LR:
        return binary_cross_entropy_loss(
            logits=logits,
            labels=labels)
    else:
        return cross_entropy_loss(
            logits=logits,
            labels=labels,
            num_classes=num_classes)


def compute_metrics(
    index: ModelIndex,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    if index == ModelIndex.STACKOVERFLOW_NWP:
        raise NotImplementedError
    elif index == ModelIndex.STACKOVERFLOW_LR:
        num_classes = tff.simulation.baselines.stackoverflow.DEFAULT_TAG_VOCAB_SIZE
        probs = jax.nn.sigmoid(logits)
        results_dict = {}
        for metric in [
                tf.keras.metrics.Precision(
                    name="precision"),
                tf.keras.metrics.Recall(
                    name="recall"),
                tf.keras.metrics.Recall(
                    top_k=5,
                    name="recall_at_5"),
                tfa.metrics.F1Score(
                    num_classes=num_classes,
                    average="micro",
                    threshold=0.5,
                    name="micro_f1"),
                tfa.metrics.F1Score(
                    num_classes=num_classes,
                    average="macro",
                    threshold=0.5,
                    name="macro_f1"),
                tfa.metrics.F1Score(
                    num_classes=num_classes,
                    average="micro",
                    name="micro_f1_at_1"),
                tfa.metrics.F1Score(
                    num_classes=num_classes,
                    average="macro",
                    name="macro_f1_at_1")]:
            metric.update_state(
                y_true=np.array(labels),
                y_pred=np.array(probs))
            results_dict[metric.name] = metric.result().numpy()
        return results_dict
    else:
        corrects = (logits.argmax(axis=-1) == labels)
        accuracy = jnp.mean(corrects, axis=0)
        return {"accuracy": accuracy}


# def apply(
#     index: ModelIndex,
#     params: jnp.ndarray,
#     rng: jax.random.KeyArray,
#     Xs: jnp.ndarray,
#     Ys: jnp.ndarray,
#     is_training: bool,
# ) -> jnp.ndarray:
#     return jax.lax.cond(
#         is_training,
#         apply_for_train,
#         apply_for_eval,
#         index,
#         params,
#         rng,
#         Xs,
#         Ys,
#     )
