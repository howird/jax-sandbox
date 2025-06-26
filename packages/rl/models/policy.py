import jax
import jax.numpy as jnp
import equinox as eqx


from jaxtyping import Array, Float


class DeterministicActor(eqx.Module):
    action_scale: Float[Array, "..."]
    action_bias: Float[Array, "..."]

    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, action_scale, action_bias, key):
        self.action_scale = action_scale
        self.action_bias = action_bias

        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim, 256, key=keys[0])
        self.layer2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.layer3 = eqx.nn.Linear(256, action_dim, key=keys[2])

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = jax.nn.tanh(self.layer3(x))
        x = x * self.action_scale + self.action_bias

        return x

    @jax.jit
    def forward_batch(self, x):
        result = jax.vmap(self.__call__)(x)
        return result
