import dataclasses
import typing as tp
import jax.numpy as jnp
from flax import nnx

@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  kv: str | None = None
  vocab: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)

@dataclasses.dataclass
class TransformerConfig:
    vocab_size: int
    emb_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    qkv_dim: int = 256
    mlp_dim: int = 1024
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    max_len: int = 128
    dtype = jnp.bfloat16
    kernel_init: nnx.Initializer = nnx.initializers.xavier_uniform()
    bias_init: nnx.Initializer = nnx.initializers.normal(stddev=1e-6)
    axis_rules: MeshRules = MeshRules(
        embed='fsdp',
        mlp='tensor',
        kv='tensor',
        vocab='tensor',
    )


@dataclasses.dataclass
class TrainerConfig:
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 0.0016
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    output_dir: str = './output'