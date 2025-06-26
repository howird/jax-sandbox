from typing import TYPE_CHECKING, Callable

import equinox as eqx
import gymnasium as gym

from equinox import AbstractVar
from jaxtyping import Array, Float, Shaped
from gymnasium.spaces.space import Space

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


class BasePolicy(eqx.Module):
    state_space: AbstractClassVar[Space]
    action_space: AbstractClassVar[Space]
    is_deterministic: AbstractClassVar[bool]
    network: Callable


# class Greedy(BasePolicy):
#     def act(
#         self, s: Float[Array, "{self.state_space}"]
#     ) -> Shaped[Array, "{self.action_space}"]:
#         pass
