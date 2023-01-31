import jax
import jaxlib
import cloudpickle
from typing import Any, Optional


# https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
def dump(obj: Any, file_name: str) -> None:
    with open(file_name, "wb") as handle:
        cloudpickle.dump(obj, handle)


def load(file_name: str) -> Any:
    with open(file_name, "rb") as handle:
        return cloudpickle.load(handle)


def get_device(device_name: Optional[str] = None) -> jaxlib.xla_extension.Device:
    devices = jax.devices(device_name)
    if len(devices) != 1:
        raise ValueError
    return devices[0]


class FakeStdin:
    def readline(self):
        return input()


def jax_jupyter_breakpoint() -> None:
    """https://github.com/google/jax/issues/11880"""
    jax.debug.breakpoint(
        backend="cli",
        stdin=FakeStdin())
