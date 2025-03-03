import typing as tp
import jax.numpy as jnp
from flax import nnx
import numpy as np
import jax
from jax import lax


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  def init(key, shape, dtype=np.float32):
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init

def shift_right(x: jax.Array, axis: int = 1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths: list[tuple[int, int]] = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
    x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
  )
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


class AddPositionalEmbs(nnx.Module):
    def __init__(self, config, *, decode=False):
        self.config = config
        self.decode = decode
        self.init_func = sinusoidal_init(config.max_len)

    def __call__(self, inputs, rngs):
        length = inputs.shape[1]
        pos_embedding = self.init_func(rngs.params(), (self.config.max_len, self.config.emb_dim))

        if self.decode:
            _, _, df = pos_embedding.shape
            pos_embedding = lax.dynamic_slice(
                pos_embedding, jnp.array((0, self.cache_index.value, 0)), (1, 1, df)
            )
        else:
            pos_embedding = pos_embedding[:, :length, :]

        return inputs + pos_embedding
    
    def init_cache(self, input_shape, dtype = jnp.float32):
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.uint32))


class MlpBlock(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
        self.linear1 = nnx.Linear(
            config.emb_dim,
            config.mlp_dim, 
            dtype=config.dtype,
            kernel_init=nnx.with_partitioning(
                config.kernel_init,
                config.axis_rules('embed', 'mlp')
            ),
            rngs=rngs)
        self.linear2 = nnx.Linear(
            config.mlp_dim,
            config.emb_dim,
            dtype=config.dtype,
            kernel_init=nnx.with_partitioning(
                config.kernel_init,
                config.axis_rules('embed', 'mlp')
            ),
            rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.dropout_rate)

    def __call__(self, inputs, rngs):
        x = self.linear1(inputs)
        x = nnx.relu(x)
        x = self.dropout(x, rngs=rngs)
        output = self.linear2(x)
        output = self.dropout(output, rngs=rngs)
        return output
    


class EncoderDecoderBlock(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
        self.ln1 = nnx.LayerNorm(
            num_features=config.emb_dim, 
            dtype=config.dtype,
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                config.axis_rules('embed')
            ),
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                config.axis_rules('embed')
            ),
            rngs=rngs)
        self.ln2 = nnx.LayerNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                config.axis_rules('embed'),
            ),
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                config.axis_rules('embed'),
            ),
            rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.emb_dim,
            qkv_features=config.qkv_dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            dtype=config.dtype,
            kernel_init=nnx.with_partitioning(
                config.kernel_init, config.axis_rules('embed', 'kv')
            ),
            bias_init=nnx.with_partitioning(
                config.bias_init, config.axis_rules('embed')
            ),
            rngs=rngs,
            )
        self.mlp = MlpBlock(config=config, rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.dropout_rate)
        
    def __call__(self, inputs, rngs, decoder_mask):
        x = self.ln1(inputs)
        x = self.attention(x, rngs=rngs, mask=decoder_mask)
        x = self.dropout(x, rngs=rngs)
        x = x + inputs
        z = self.ln2(x)
        z = self.mlp(z, rngs)

        return z

class Decoder(nnx.Module):
    def __init__(self, config, *, decode = False, rngs):
        self.config = config
        self.decode = decode
        self.output_embed = nnx.Embed(num_embeddings=config.vocab_size, 
                                      features=config.emb_dim,
                                      embedding_init=nnx.with_partitioning(
                                            nnx.initializers.normal(stddev=1.0),
                                            config.axis_rules('vocab', 'embed'),
                                        ),
                                      rngs=rngs)
        self.posembed_out = AddPositionalEmbs(config=config)
        self.dropout = nnx.Dropout(rate=config.dropout_rate)

        for idx in range(config.num_layers):
            layer = EncoderDecoderBlock(config,rngs)
            setattr(self, f'encoderdecoderblock_{idx}', layer)

        self.encoderdecoder_norm = nnx.LayerNorm(
            num_features=config.emb_dim, 
            dtype=config.dtype,
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), config.axis_rules('embed')
            ),
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(), config.axis_rules('embed')
            ),
            rngs=rngs)

    def __call__(self, inputs, rngs, decoder_mask=None):
        y = inputs.astype('int32')
        if not self.decode:
            y = shift_right(y)
        y = self.output_embed(y)
        y = self.posembed_out(y, rngs=rngs)
        y = self.dropout(y, rngs=rngs)

        y = y.astype(self.config.dtype)

        for idx in range(self.config.num_layers):
            layer = getattr(self, f'encoderdecoderblock_{idx}')
            y = layer(y, rngs=rngs, decoder_mask=decoder_mask)

        y = self.encoderdecoder_norm(y)

        logits = self.output_embed.attend(y)
        logits = logits / jnp.sqrt(y.shape[-1])
        return logits


class TransformerLM(nnx.Module):
    def __init__(self, config, *, decode=False, rngs):
        self.config = config
        self.decode = decode
        self.decoder = Decoder(config, rngs=rngs)

    def __call__(self, inputs, rngs):
        if self.decode:
            decoder_mask = None
        else:
            decoder_mask = nnx.combine_masks(
                nnx.make_attention_mask(inputs > 0, inputs > 0, dtype=self.config.dtype),
                nnx.make_causal_mask(inputs, dtype=self.config.dtype),
            )
            
        logits = self.decoder(inputs, rngs, decoder_mask=decoder_mask)

        return logits.astype(self.config.dtype)