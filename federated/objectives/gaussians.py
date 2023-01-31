import jax
import attr
import logging
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Optional
from tensorflow_probability.substrates import jax as tfp

from ..utils import misc_utils
from .gaussian_types import (
    Covariance,
    Precision,
)


# https://github.com/tensorflow/probability/issues/1523
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())


class Gaussian(object):

    def __init__(
            self,
            mu: Optional[jnp.ndarray] = None,
            eta: Optional[jnp.ndarray] = None,
            Sigma: Optional[Covariance] = None,
            Lambda: Optional[Precision] = None,
            check_params: bool = True,
    ) -> None:
        # only one pair is allowed
        if (mu is None) == (eta is None):
            raise ValueError
        if (Sigma is None) == (Lambda is None):
            raise ValueError

        self._mu = mu
        self._eta = eta
        self._Sigma = Sigma
        self._Lambda = Lambda
        if check_params:
            self._check_params()

    def _check_params(self) -> None:
        raise NotImplementedError

    @property
    def mu(self) -> jnp.ndarray:
        raise NotImplementedError

    @property
    def Sigma(self) -> Covariance:
        raise NotImplementedError

    @property
    def eta(self) -> jnp.ndarray:
        raise NotImplementedError

    @property
    def Lambda(self) -> Precision:
        raise NotImplementedError

    @classmethod
    def from_products(
            cls,
            qs: List["Gaussian"],
    ) -> "Gaussian":
        raise NotImplementedError

    @classmethod
    def from_quotient(
            cls,
            q1: "Gaussian",
            q2: "Gaussian",
    ) -> "Gaussian":
        raise NotImplementedError

    @classmethod
    def from_power(
            cls,
            q: "Gaussian",
            power: float,
    ) -> "Gaussian":
        raise NotImplementedError

    def damped_updates(
            self,
            qs: List["Gaussian"],
            delta: float,
    ) -> "Gaussian":
        raise NotImplementedError

    @classmethod
    def check_types_compatible(
            cls,
            qs: List["Gaussian"],
    ) -> None:
        type_checks = [type(q) is cls for q in qs]
        if not all(type_checks):
            raise TypeError(f"All elements of `qs` must be of type "
                            f"{cls}, but got {type_checks}")

    @staticmethod
    @jax.jit
    def compute_nll(
            x: jnp.ndarray,
            eta: jnp.ndarray,
            Lambda: Precision,
    ) -> jnp.ndarray:
        raise NotImplementedError

    def nll(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.compute_nll(
            x=x,
            eta=self.eta,
            Lambda=self.Lambda)

    def _get_string_form(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self._get_string_form()

    def __str__(self) -> str:
        return self._get_string_form()
    
    def to(self, device_name: str) -> "Gaussian":
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def compute_sample(
            prng_key: jax.random.KeyArray,
            mu: jnp.ndarray,
            Sigma: Covariance,
    ) -> jnp.ndarray:
        raise NotImplementedError


class SimpleGaussian(Gaussian):
    """Gaussian distribution

    One parameterization:
        mu, Sigma
    Another parameterization (preferred):
        eta, Lambda
    """

    def __init__(
            self,
            mu: Optional[jnp.ndarray] = None,
            eta: Optional[jnp.ndarray] = None,
            Sigma: Optional[jnp.ndarray] = None,
            Lambda: Optional[jnp.ndarray] = None,
            check_params: bool = True,
    ) -> None:
        # only one pair is allowed
        if (mu is None) == (eta is None):
            raise ValueError
        if (Sigma is None) == (Lambda is None):
            raise ValueError

        self._mu = mu
        self._eta = eta
        self._Sigma = Sigma
        self._Lambda = Lambda
        if check_params:
            self._check_params()

    def _check_params(self) -> None:
        # only check `eta` and `Lambda`
        d = self.eta.shape[0]
        checks = [
            (self.eta.ndim == 1),
            (self.Lambda.ndim == 2),
            (self.Lambda.shape == (d, d))]

        if not all(checks):
            raise ValueError(checks)

    @property
    def mu(self) -> jnp.ndarray:
        if self._mu is not None:
            return self._mu
        return jnp.linalg.inv(self._Lambda) @ self._eta

    @property
    def Sigma(self) -> jnp.ndarray:
        if self._Sigma is not None:
            return self._Sigma
        return jnp.linalg.inv(self._Lambda)

    @property
    def eta(self) -> jnp.ndarray:
        if self._eta is not None:
            return self._eta
        return jnp.linalg.inv(self._Sigma) @ self._mu

    @property
    def Lambda(self) -> jnp.ndarray:
        if self._Lambda is not None:
            return self._Lambda
        return jnp.linalg.inv(self._Sigma)

    @classmethod
    def from_products(
            cls,
            qs: List["SimpleGaussian"],
    ) -> "SimpleGaussian":
        cls.check_types_compatible(qs)
        new_eta = qs[0].eta
        new_Lambda = qs[0].Lambda
        for q in qs[1:]:
            new_eta = new_eta + q.eta
            new_Lambda = new_Lambda + q.Lambda

        return cls(
            eta=new_eta,
            Lambda=new_Lambda)

    @classmethod
    def from_quotient(
            cls,
            q1: "SimpleGaussian",
            q2: "SimpleGaussian",
    ) -> "SimpleGaussian":
        cls.check_types_compatible([q1, q2])
        return cls(
            eta=q1.eta - q2.eta,
            Lambda=q1.Lambda - q2.Lambda)

    @classmethod
    def from_power(
            cls,
            q: "SimpleGaussian",
            power: float,
    ) -> "SimpleGaussian":
        cls.check_types_compatible([q])
        return cls(
            eta=power * q.eta,
            Lambda=power * q.Lambda)

    @staticmethod
    @jax.jit
    def compute_nll(
            x: jnp.ndarray,
            eta: jnp.ndarray,
            Lambda: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of a Gaussian distribution,
        up to some constants that do not depend on the `x`."""
        # When `x` and `eta` are vectors, the transposes have no
        # effect, but they are still correct, so we will leave them here.
        return 0.5 * x.T @ Lambda @ x - eta @ x

    def nll(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.compute_nll(
            x=x,
            eta=self.eta,
            Lambda=self.Lambda)

    def _get_string_form(self) -> str:
        mu_string = "none"
        eta_string = "none"
        Sigma_string = "none"
        Lambda_string = "none"
        if self._mu is not None:
            mu_string = f"{self._mu.shape}"

        if self._eta is not None:
            eta_string = f"{self._eta.shape}"

        if self._Sigma is not None:
            Sigma_string = f"{self._Sigma.shape}"

        if self._Lambda is not None:
            Lambda_string = f"{self._Lambda.shape}"

        return (
            f"{self.__class__.__name__}"
            f"(\u03BC={mu_string}, \u03B7={eta_string}, "
            f"\u03A3={Sigma_string}, \u039B={Lambda_string})")

    def to(self, device_name: str) -> "Gaussian":

        def _maybe_to(x: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
            if x is None:
                return None
            device = misc_utils.get_device(device_name)
            return jax.device_put(x, device)

        return self.__class__(
            mu=_maybe_to(self._mu),
            eta=_maybe_to(self._eta),
            Sigma=_maybe_to(self._Sigma),
            Lambda=_maybe_to(self._Lambda),
        )


class DiagonalGaussian(SimpleGaussian):

    def _check_params(self) -> None:
        # only check `eta` and `Lambda`
        d = self.eta.shape[0]
        checks = [
            (self.eta.ndim == 1),
            (self.Lambda.ndim == 1),
            (self.Lambda.shape[0] == d)]

        if not all(checks):
            raise ValueError(checks)

    @property
    def mu(self) -> jnp.ndarray:
        if self._mu is not None:
            return self._mu
        return self._eta / self._Lambda

    @property
    def Sigma(self) -> jnp.ndarray:
        if self._Sigma is not None:
            return self._Sigma
        return 1. / self._Lambda

    @property
    def eta(self) -> jnp.ndarray:
        if self._eta is not None:
            return self._eta
        return self._mu / self._Sigma

    @property
    def Lambda(self) -> jnp.ndarray:
        if self._Lambda is not None:
            return self._Lambda
        return 1. / self._Sigma

    @staticmethod
    @jax.jit
    def compute_nll(
            x: jnp.ndarray,
            eta: jnp.ndarray,
            Lambda: jnp.ndarray,
    ) -> jnp.ndarray:
        return 0.5 * x * Lambda @ x - eta @ x

    def is_proper(self) -> bool:
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def compute_sample(
            prng_key: jax.random.KeyArray,
            mu: jnp.ndarray,
            Sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        # We need standard deviations here, hence `sqrt(Sigma)`
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=jnp.sqrt(Sigma))
        return dist.sample(seed=prng_key)


def test_gaussian() -> None:
    key = jax.random.PRNGKey(5)
    key, subkey = jax.random.split(key)

    num_units = 17
    num_gaussians = 7

    q1s = []
    q2s = []
    for _ in range(num_gaussians):
        key, subkey = jax.random.split(key)
        mu = jax.random.normal(
            key=subkey,
            shape=[num_units])
        key, subkey = jax.random.split(key)
        Sigma = jax.random.normal(
            key=subkey,
            shape=[num_units, num_units])
        q1s.append(SimpleGaussian(mu=mu, Sigma=Sigma))

    for _ in range(num_gaussians):
        key, subkey = jax.random.split(key)
        eta = jax.random.normal(
            key=subkey,
            shape=[num_units])
        key, subkey = jax.random.split(key)
        Lambda = jax.random.normal(
            key=subkey,
            shape=[num_units, num_units])
        q2s.append(SimpleGaussian(eta=eta, Lambda=Lambda))

    print("Check parameters")
    for q in q1s + q2s:
        _mu = jnp.matmul(jnp.linalg.inv(q.Lambda), q.eta)
        _eta = jnp.matmul(jnp.linalg.inv(q.Sigma), q.mu)
        _Sigma = jnp.linalg.inv(q.Lambda)
        _Lambda = jnp.linalg.inv(q.Sigma)
        passes = [
            jnp.allclose(q.mu, _mu, atol=1e-5).item(),
            jnp.allclose(q.eta, _eta, atol=1e-5).item(),
            jnp.allclose(q.Sigma, _Sigma, atol=1e-5).item(),
            jnp.allclose(q.Lambda, _Lambda, atol=1e-5).item(),
        ]

        _Sigma_cond = jnp.linalg.cond(q.Sigma)
        _Lambda_cond = jnp.linalg.cond(q.Lambda)
        print(f"{(passes)} \t "
              f"cond(Sigma): {_Sigma_cond:10.3f} \t "
              f"cond(Lambda): {_Lambda_cond:10.3f}")

    print("\nCheck products")
    for qs in [q1s, q2s, q1s + q2s]:
        qs_combined = SimpleGaussian.from_products(qs)
        _Sigma = jnp.linalg.inv(jnp.sum(jnp.stack(
            [jnp.linalg.inv(q.Sigma) for q in qs], axis=0), axis=0))
        _Lambda = jnp.sum(jnp.stack(
            [q.Lambda for q in qs], axis=0), axis=0)
        _mu = jnp.matmul(_Sigma, jnp.sum(jnp.stack(
            [jnp.matmul(jnp.linalg.inv(q.Sigma), q.mu) for q in qs], axis=0), axis=0))
        _eta = jnp.sum(jnp.stack(
            [q.eta for q in qs], axis=0), axis=0)

        passes = [
            jnp.allclose(qs_combined.mu, _mu, atol=7e-5).item(),
            jnp.allclose(qs_combined.eta, _eta, atol=1e-5).item(),
            jnp.allclose(qs_combined.Sigma, _Sigma, atol=5e-5).item(),
            jnp.allclose(qs_combined.Lambda, _Lambda, atol=1e-5).item(),
        ]

        print(f"{(passes)}")

    print("\nCheck quotients")
    for qs in [q1s, q1s[::-1], q2s, q2s[::-1], q1s + q2s, (q1s + q2s)[::-1]]:
        _q1 = qs[0]
        _q2 = qs[-1]
        qs_divided = SimpleGaussian.from_quotient(_q1, _q2)
        _Sigma = jnp.linalg.inv(jnp.linalg.inv(_q1.Sigma) -
                                jnp.linalg.inv(_q2.Sigma))
        _Lambda = _q1.Lambda - _q2.Lambda
        _mu = jnp.matmul(_Sigma,
                         jnp.matmul(jnp.linalg.inv(_q1.Sigma), _q1.mu) -
                         jnp.matmul(jnp.linalg.inv(_q2.Sigma), _q2.mu))
        _eta = _q1.eta - _q2.eta

        passes = [
            jnp.allclose(qs_divided.mu, _mu, atol=1e-5).item(),
            jnp.allclose(qs_divided.eta, _eta, atol=1e-5).item(),
            jnp.allclose(qs_divided.Sigma, _Sigma, atol=1e-5).item(),
            jnp.allclose(qs_divided.Lambda, _Lambda, atol=1e-5).item(),
        ]

        print(f"{(passes)}")


def test_diagonal_gaussian() -> None:
    d = 97
    def get_random_x(): return jnp.array(np.random.normal(size=[d]))
    def get_random_eta(): return jnp.array(np.random.normal(size=[d]))
    def get_random_Lambda(): return jnp.array(np.random.normal(size=[d, d]))

    def _check_equality(qd: SimpleGaussian, q: SimpleGaussian) -> None:
        print((qd.eta ==
               q.eta).all(),
              "\t", jnp.linalg.norm(qd.eta),
              "\t", jnp.linalg.norm(q.eta))
        print((jnp.diagflat(qd.Lambda) ==
               q.Lambda).all(),
              "\t", jnp.linalg.norm(qd.Lambda),
              "\t", jnp.linalg.norm(q.Lambda),
              "\t", qd.Lambda.shape,
              "\t", q.Lambda.shape)

    for _ in range(3):
        x = get_random_x()
        qd1 = DiagonalGaussian(
            eta=get_random_eta(),
            Lambda=jnp.diagonal(get_random_Lambda()))
        qd2 = DiagonalGaussian(
            eta=get_random_eta(),
            Lambda=jnp.diagonal(get_random_Lambda()))
        qd3 = DiagonalGaussian(
            eta=get_random_eta(),
            Lambda=jnp.diagonal(get_random_Lambda()))
        q1 = SimpleGaussian(
            eta=qd1.eta,
            Lambda=jnp.diagflat(qd1.Lambda))
        q2 = SimpleGaussian(
            eta=qd2.eta,
            Lambda=jnp.diagflat(qd2.Lambda))
        q3 = SimpleGaussian(
            eta=qd3.eta,
            Lambda=jnp.diagflat(qd3.Lambda))

        print(jnp.allclose(qd1.nll(x), q1.nll(x)),
              "\t", qd1.nll(x),
              "\t", q1.nll(x))
        print(jnp.allclose(qd2.nll(x), q2.nll(x)),
              "\t", qd2.nll(x),
              "\t", q2.nll(x))
        print(jnp.allclose(qd3.nll(x), q3.nll(x)),
              "\t", qd3.nll(x),
              "\t", q3.nll(x))

        _check_equality(
            DiagonalGaussian.from_products([qd1, qd2, qd3]),
            SimpleGaussian.from_products([q1, q2, q3]))

        _check_equality(
            DiagonalGaussian.from_quotient(qd2, qd1),
            SimpleGaussian.from_quotient(q2, q1))

        _check_equality(
            qd3.damped_updates([qd1, qd2], delta=0.71),
            q3.damped_updates([q1, q2], delta=0.71))


def test_damped_updates() -> None:
    d = 97
    def get_random_x(): return jnp.array(np.random.normal(size=[d]))
    def get_random_eta(): return jnp.array(np.random.normal(size=[d]))
    def get_random_Lambda(): return jnp.array(np.random.normal(size=[d, d]))

    for _ in range(7):
        delta = np.random.random()
        q = SimpleGaussian(
            eta=get_random_eta(),
            Lambda=get_random_Lambda())
        qs = [
            SimpleGaussian(
                eta=get_random_eta(),
                Lambda=get_random_Lambda())
            for _ in range(11)]
        q1 = q.damped_updates(qs, delta=delta)
        q2 = SimpleGaussian(
            eta=q.eta + delta * SimpleGaussian.from_products(qs).eta,
            Lambda=q.Lambda + delta * SimpleGaussian.from_products(qs).Lambda)

        print(
            f"Norms: "
            f"{jnp.abs(q1.eta).max():<5.2f} "
            f"{jnp.abs(q2.eta).max():<5.2f} "
            f"{jnp.abs(q1.Lambda).max():<5.2f} "
            f"{jnp.abs(q2.Lambda).max():<5.2f} "
            f"| Diffs: "
            f"{jnp.abs(q1.eta -    q2.eta).max():.2e}",
            f"{jnp.abs(q1.Lambda - q2.Lambda).max():.2e}",
            f"\t{jnp.allclose(q1.eta,    q2.eta)}",
            f"\t{jnp.allclose(q1.Lambda, q2.Lambda, atol=7e-7)}")
