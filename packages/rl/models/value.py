class QNetwork(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, key):
        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim + action_dim, 256, key=keys[0])
        self.layer2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.layer3 = eqx.nn.Linear(256, 1, key=keys[2])

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    @jax.jit
    def forward_batch(self, x, a):
        result = jax.vmap(self.__call__)(x, a)
        return result
