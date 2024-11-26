import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import chex

class MLP(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(self, rng: chex.PRNGKey, layer_shapes: list[int]):
        """
        Creates a MLP with given layer dimensions.

        :param list[int] layer_shapes: Shape of each layer, with the following pattern: [input_layer, hidden_layer_1, ..., output_layer]
        """
        ### ------------------------- To implement -------------------------
        layers = []
        keys = jax.random.split(rng, len(layer_shapes) - 1)

        for i in range(len(layer_shapes) - 1):
            linear_layer = eqx.nn.Linear(layer_shapes[i], layer_shapes[i + 1], key=keys[i])

            activation_layer = eqx.nn.Lambda(jax.nn.relu)

            layers.append(linear_layer)
            layers.append(activation_layer)

        if len(layers) > 0 and isinstance(layers[-1], eqx.nn.Lambda):
            layers.pop()

        self.layers = eqx.nn.Sequential(layers)
  

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layers(x)


class CNN_2048(eqx.Module):
    base_block: eqx.Module
    conv_block_1: eqx.Module
    max_pool_1: eqx.Module
    conv_block_2: eqx.Module
    max_pool_2: eqx.Module
    mlp: eqx.Module

    def __init__(self, rng: chex.PRNGKey, num_outputs: int):
        super().__init__()
        rng, key_cnn_1, key_cnn_2, key_cnn_3, key_cnn_4, key_cnn_5 = jax.random.split(rng, 6)
        self.base_block = nn.Conv2d(31, 64, 2, padding="SAME", key=key_cnn_1)   # (4x4x31) -> (4x4x64)

        self.conv_block_1 = nn.Sequential([
            nn.Conv2d(64, 64, 2, padding="SAME", key=key_cnn_2),    # (4x4x64) -> (4x4x64)
            nn.LayerNorm((64, 4, 4)),
            nn.Conv2d(64, 64, 2,  padding="SAME", key=key_cnn_3),   # (4x4x64) -> (4x4x64)
            nn.Lambda(jax.nn.gelu)
        ])
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)                 # (4x4x64) -> (2x2x64)

        self.conv_block_2 = nn.Sequential([
            nn.Conv2d(64, 64, 2, padding="SAME", key=key_cnn_4),    # (2x2x64) -> (2x2x64)
            nn.LayerNorm((64, 2, 2)),
            nn.Conv2d(64, 64, 2,  padding="SAME", key=key_cnn_5),   # (2x2x64) -> (2x2x64)
            nn.Lambda(jax.nn.gelu)
        ])
        self.max_pool_2 = nn.MaxPool2d(2, stride=2)                 # (2x2x64) -> (1x1x64)
        self.mlp = MLP(rng, [64, 64, num_outputs])

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        x_base = self.base_block(x.transpose((2, 0, 1)).astype(jnp.float32))
        x1_conv = self.conv_block_1(x_base)
        x1 = self.max_pool_1(x1_conv + x_base)

        x2_conv = self.conv_block_2(x1)
        x2 = self.max_pool_2(x2_conv + x1)
        return self.mlp(x2.flatten())
