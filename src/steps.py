from .trainer import (
    TrainState,
    compute_weighted_cross_entropy,
    compute_metrics
)
import jax.numpy as jnp
import jax
from flax import nnx
from .temperature_sampler import temperature_sample

def train_step(
  state: TrainState,
  batch,
  learning_rate_fn,
  label_smoothing=0.0,
  dropout_rng=None,
):
  """Perform a single training step."""
  
  inner = batch
  weights = jnp.where(inner > 0, 1, 0).astype(jnp.float32)

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    module = nnx.merge(state.graphdef, params, state.keys, state.rest)
    module.set_attributes(deterministic=False, decode=False)
    logits = module(
      inner,
      rngs=nnx.Rngs(dropout=dropout_rng, default=dropout_rng),
    )

    loss, weight_sum = compute_weighted_cross_entropy(
      logits, inner, weights, label_smoothing
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, inner, weights)
  metrics['learning_rate'] = lr

  return new_state, metrics

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class HasCache(Protocol):
  def init_cache(self, input_shape, dtype = jnp.float32): ...

def eval_step(
  state: TrainState,
  batch,
  label_smoothing=0.0,
):
  """Calculate evaluation metrics on a batch."""
  weights = jnp.where(batch > 0, 1.0, 0.0)
  module = nnx.merge(state.graphdef, state.params, state.keys, state.rest)
  module.set_attributes(deterministic=True, decode=False)
  logits = module(batch, rngs=nnx.Rngs(0))

  return compute_metrics(logits, batch, weights, label_smoothing)


def predict_step(
  inputs,
  params: nnx.State,
  rngkey: jax.Array,
  graphdef: nnx.GraphDef,
  eos_id: int,
  max_decode_len: int,
  config,
  temperature: float,
  top_k: int,
):
  """Predict language model on a batch."""
  module = nnx.merge(graphdef, params)

  # TODO(cgarciae): check how pytorch does this.
  for _path, m in module.iter_modules():
    if isinstance(m, HasCache):
      input_shape = (inputs.shape[0], max_decode_len, config.emb_dim)
      m.init_cache(input_shape, dtype=config.dtype)

  graphdef, params, cache = nnx.split(module, nnx.Param, nnx.Cache)

  def tokens_ids_to_logits(flat_ids, cache: nnx.State):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    module = nnx.merge(graphdef, params, cache)
    module.set_attributes(deterministic=True, decode=True)
    logits = module(flat_ids)
    cache = nnx.state(module, nnx.Cache)
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    logits = logits.squeeze(axis=1)
    return logits, cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  seqs = temperature_sample(
    inputs,
    cache,
    tokens_ids_to_logits,
    rngkey,
    temperature=temperature,
    topk=top_k,
    eos_token=eos_id,
  )

  return seqs