import attr
import functools
import jax
import jax.nn as jnn
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict, Any

from . import gaussians
from ..modules import utils as modules_utils


class Dataset(NamedTuple):
    Xs: jnp.ndarray
    Ys: jnp.ndarray


@attr.s(eq=False)
class SimpleObjective(object):

    model_index: modules_utils.ModelIndex = attr.ib()
    Xs: jnp.ndarray = attr.ib()
    Ys: jnp.ndarray = attr.ib()
    batch_size: int = attr.ib()
    num_epochs: int = attr.ib()
    num_classes: int = attr.ib()

    @property
    def data(self) -> Dataset:
        return Dataset(Xs=self.Xs, Ys=self.Ys)

    @property
    def num_points(self) -> int:
        """Number of points in the dataset, per epoch."""
        num_total_points = self.Xs.shape[0]
        if num_total_points % self.num_epochs != 0:
            raise ValueError

        return int(num_total_points / self.num_epochs)

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {}

    @property
    def dim(self) -> int:
        return modules_utils.get_dim(self.model_index)

    def init(self, prng_key: jax.random.KeyArray) -> jnp.ndarray:
        return modules_utils.init(
            index=self.model_index,
            rng=prng_key,
            Xs=self.Xs,
            Ys=self.Ys)

    def evaluate(
        self,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
    ) -> Dict[str, jnp.ndarray]:
        logits, loss = self.compute_evaluation(
            model_index=self.model_index,
            params=params,
            prng_key=prng_key,
            data_batch=self.data,
            num_classes=self.num_classes,
            num_points=self.num_points,
        )

        metrics = modules_utils.compute_metrics(
            index=self.model_index,
            logits=logits,
            labels=self.data.Ys,
        )
        metrics["loss"] = loss
        return metrics

    def evaluate_in_batches(
        self,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        cutoff: int,
    ) -> Dict[str, jnp.ndarray]:

        def _compute_evaluation(data_batch: Dataset) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # We use the same key here. Are there cases
            # in which we need to split the key again?
            return self.compute_evaluation(
                model_index=self.model_index,
                params=params,
                prng_key=prng_key,
                data_batch=data_batch,
                num_classes=self.num_classes,
                num_points=self.num_points,
            )

        indices_1 = jnp.arange(0, cutoff)
        indices_2 = jnp.arange(cutoff, self.data.Xs.shape[0])
        batch_size_1 = indices_1.shape[0]
        batch_size_2 = indices_2.shape[0]
        data_1 = jax.tree_util.tree_map(
            lambda A: jnp.take(A, indices_1, axis=0),
            self.data)
        data_2 = jax.tree_util.tree_map(
            lambda A: jnp.take(A, indices_2, axis=0),
            self.data)

        logits_1, loss_1 = _compute_evaluation(data_1)
        logits_2, loss_2 = _compute_evaluation(data_2)

        logits = jnp.concatenate([logits_1, logits_2], axis=0)
        loss = (
            loss_1 * batch_size_1 +
            loss_2 * batch_size_2
        ) / (batch_size_1 + batch_size_2)

        metrics = modules_utils.compute_metrics(
            index=self.model_index,
            logits=logits,
            labels=self.data.Ys,
        )
        metrics["loss"] = loss
        return metrics


    def nll(
        self,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        return self.compute_nll_for_train(
            model_index=self.model_index,
            params=params,
            prng_key=prng_key,
            data_batch=self.data,
            num_classes=self.num_classes,
            num_points=self.num_points,
            **self.kwargs,
        )

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_nll_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
    ) -> jnp.ndarray:
        """
        We use the sum loss here instead of average loss. This is because
        HMC samplers require joint probabilities.

        In a non-stochastic setting, the sum-loss is,

            loss = [ sum_{all} -log p(x|theta) ] - log p(theta),

        where the prior is handled later. Hence, in the stochastic setting,
        we should have the sum-loss,

            loss = N/B [ sum_{batch} -log p(x|theta) ] - log p(theta),

        where N is the number of data points and B is the batch size.
        """

        if params.ndim != 1:
            raise ValueError("x must be a vector.")

        batch_size = data_batch.Xs.shape[0]
        logits = modules_utils.apply_for_train(
            index=model_index,
            params=params,
            rng=prng_key,
            Xs=data_batch.Xs,
            Ys=data_batch.Ys,
        )
        losses = modules_utils.compute_loss(
            index=model_index,
            logits=logits,
            labels=data_batch.Ys,
            num_classes=num_classes,
        )
        batch_sum_loss = jnp.sum(losses, axis=0)
        return num_points / batch_size * batch_sum_loss

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_loss_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
        **kwargs,
    ) -> jnp.ndarray:

        sum_loss = cls.compute_nll_for_train(
            model_index=model_index,
            params=params,
            prng_key=prng_key,
            data_batch=data_batch,
            num_classes=num_classes,
            num_points=num_points,
            **kwargs,
        )
        return sum_loss / num_points

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_grad_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
        **kwargs,
    ) -> jnp.ndarray:

        def _loss_computation(_params: jnp.ndarray) -> jnp.ndarray:
            return cls.compute_loss_for_train(
                model_index=model_index,
                params=_params,
                prng_key=prng_key,
                data_batch=data_batch,
                num_classes=num_classes,
                num_points=num_points,
                **kwargs,
            )

        return jax.grad(_loss_computation)(params)

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_evaluation(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        if params.ndim != 1:
            raise ValueError("x must be a vector.")

        logits = modules_utils.apply_for_eval(
            index=model_index,
            params=params,
            rng=prng_key,
            Xs=data_batch.Xs,
            Ys=data_batch.Ys,
        )
        losses = modules_utils.compute_loss(
            index=model_index,
            logits=logits,
            labels=data_batch.Ys,
            num_classes=num_classes,
        )
        loss = jnp.mean(losses, axis=0)
        return logits, loss

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_evaluation_in_batches(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data: Dataset,
        num_classes: int,
        num_points: int,
        batch_size: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise ValueError
        if data.Xs.shape[0] != num_points:
            raise ValueError

        def _batch_computation(
            unused_carry: None,
            data_batch: Dataset,
        ) -> Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]:
            if data_batch.Xs.shape[0] != batch_size:
                raise ValueError

            loss, accuracy = cls.compute_evaluation(
                model_index=model_index,
                params=params,
                prng_key=prng_key,
                data_batch=data_batch,
                num_classes=num_classes,
                num_points=num_points)

            losses = loss * batch_size
            corrects = accuracy * batch_size
            return None, (losses, corrects)

        data_batches = cls.generate_data_batches(
            prng_key=prng_key,
            data=data,
            batch_size=batch_size)
        carry, (losses, corrects) = jax.lax.scan(
            _batch_computation,
            None,
            data_batches)
        loss = jnp.sum(losses, axis=0) / num_points
        accuracy = jnp.sum(corrects, axis=0) / num_points
        return loss, accuracy

    @classmethod
    def generate_data_batches(
        cls,
        prng_key: jax.random.KeyArray,
        data: Dataset,
        batch_size: int,
    ) -> Dataset:

        num_points = data.Xs.shape[0]
        # if num_points % batch_size != 0:
        #     raise ValueError

        num_batches = int(num_points / batch_size)
        indices = jax.random.choice(
            prng_key,
            num_points,
            shape=(num_batches, batch_size),
            replace=False)
        data_batches = jax.tree_util.tree_map(
            lambda A: jnp.take(A, indices, axis=0),
            data)

        return data_batches

    @classmethod
    def generate_data_epochs(
        cls,
        data: Dataset,
        num_epochs: int,
        num_points: int,
    ) -> Dataset:

        if data.Xs.shape[0] != num_epochs * num_points:
            raise ValueError

        indices = (
            jnp
            .arange(num_epochs * num_points)
            .reshape(num_epochs, num_points))
        data_batches = jax.tree_util.tree_map(
            lambda A: jnp.take(A, indices, axis=0),
            data)

        return data_batches

    @classmethod
    def sample_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
    ) -> jnp.ndarray:
        subkey_0, subkey_1 = jax.random.split(prng_key)
        logits = modules_utils.apply_for_train(
            index=model_index,
            params=params,
            rng=subkey_0,
            Xs=data_batch.Xs,
            Ys=data_batch.Ys,
        )
        return jax.random.categorical(subkey_1, logits, axis=-1)

    @classmethod
    def compute_Fisher_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
        reduce_op: str,
    ) -> jnp.ndarray:

        def fisher_fn(
            key: jax.random.KeyArray,
            data_example: Dataset,
        ) -> jnp.ndarray:
            subkey_s, subkey_g = jax.random.split(key)
            sampled_Ys = cls.sample_for_train(
                model_index=model_index,
                params=params,
                prng_key=subkey_s,
                data_batch=data_example)
            sampled_data_example = Dataset(
                Xs=data_example.Xs,
                Ys=sampled_Ys)

            # We only approximate the data-NLL with Fisher, hence
            # we should not include the loss from the priors.
            grads = SimpleObjective.compute_grad_for_train(
                model_index=model_index,
                params=params,
                prng_key=subkey_g,
                data_batch=sampled_data_example,
                num_classes=num_classes,
                num_points=num_points,
                # **objective_kwargs,
            )

            return grads ** 2

        subkeys = jax.random.split(
            prng_key,
            data_batch.Xs.shape[0])

        # [num_examples, ...] --> [num_examples, 1, ...]
        data_batch = jax.tree_util.tree_map(
            lambda A: jnp.expand_dims(A, axis=1),
            data_batch)

        # [num_examples, dim]
        Fisher = jax.vmap(fisher_fn)(subkeys, data_batch)

        if reduce_op == "MEAN":
            return Fisher.mean(axis=0)
        else:
            raise NotImplementedError


@attr.s(eq=False)
class ObjectiveWithGaussianPrior(SimpleObjective):

    prior_eta: jnp.ndarray = attr.ib()
    prior_Lambda: jnp.ndarray = attr.ib()
    prior_strength: float = attr.ib(default=1.0)

    @classmethod
    def from_objective(
            cls,
            objective: SimpleObjective,
            prior: gaussians.SimpleGaussian,
            prior_strength: float = 1.0,
    ) -> "ObjectiveWithGaussianPrior":
        if type(objective) is not SimpleObjective:
            raise TypeError
        if type(prior) is not gaussians.SimpleGaussian:
            raise TypeError

        return cls(
            model_index=objective.model_index,
            Xs=objective.Xs,
            Ys=objective.Ys,
            batch_size=objective.batch_size,
            num_epochs=objective.num_epochs,
            num_classes=objective.num_classes,
            prior_eta=prior.eta,
            prior_Lambda=prior.Lambda,
            prior_strength=prior_strength,
        )

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {
            "prior_eta": self.prior_eta,
            "prior_Lambda": self.prior_Lambda,
            "prior_strength": self.prior_strength,
        }

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_nll_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
        prior_eta: jnp.ndarray,
        prior_Lambda: jnp.ndarray,
        prior_strength: float,
    ) -> jnp.ndarray:
        likelihood_loss = super().compute_nll_for_train(
            model_index=model_index,
            params=params,
            prng_key=prng_key,
            data_batch=data_batch,
            num_classes=num_classes,
            num_points=num_points)
        prior_loss = gaussians.SimpleGaussian.compute_nll(
            x=params,
            eta=prior_eta,
            Lambda=prior_Lambda)

        return likelihood_loss + prior_strength * prior_loss


@attr.s(eq=False)
class ObjectiveWithDiagGaussianPrior(SimpleObjective):

    prior_eta: jnp.ndarray = attr.ib()
    prior_diag_Lambda: jnp.ndarray = attr.ib()
    prior_strength: float = attr.ib(default=1.0)

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {
            "prior_eta": self.prior_eta,
            "prior_diag_Lambda": self.prior_diag_Lambda,
            "prior_strength": self.prior_strength,
        }

    @classmethod
    def from_objective(
            cls,
            objective: SimpleObjective,
            prior: gaussians.DiagonalGaussian,
            prior_strength: float = 1.0,
    ) -> "ObjectiveWithDiagGaussianPrior":
        if type(objective) is not SimpleObjective:
            raise TypeError
        if type(prior) is not gaussians.DiagonalGaussian:
            raise TypeError

        return cls(
            model_index=objective.model_index,
            Xs=objective.Xs,
            Ys=objective.Ys,
            batch_size=objective.batch_size,
            num_epochs=objective.num_epochs,
            num_classes=objective.num_classes,
            prior_eta=prior.eta,
            prior_diag_Lambda=prior.Lambda,
            prior_strength=prior_strength,
        )

    @classmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "cls",
            "model_index",
            "num_classes",
        ),
    )
    def compute_nll_for_train(
        cls,
        model_index: modules_utils.ModelIndex,
        params: jnp.ndarray,
        prng_key: jax.random.KeyArray,
        data_batch: Dataset,
        num_classes: int,
        num_points: int,
        prior_eta: jnp.ndarray,
        prior_diag_Lambda: jnp.ndarray,
        prior_strength: float,
    ) -> jnp.ndarray:
        likelihood_loss = super().compute_nll_for_train(
            model_index=model_index,
            params=params,
            prng_key=prng_key,
            data_batch=data_batch,
            num_classes=num_classes,
            num_points=num_points)
        prior_loss = gaussians.DiagonalGaussian.compute_nll(
            x=params,
            eta=prior_eta,
            Lambda=prior_diag_Lambda)

        return likelihood_loss + prior_strength * prior_loss
