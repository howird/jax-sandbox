import equinox as eqx

from equinox import AbstractVar

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


class RLAlgo(eqx.Module):
    pass


class ModelBasedAlgo(RLAlgo):
    pass


class ModelFreeAlgo(RLAlgo):
    pass
