from trainer import (
    TrainState,
    compute_weighted_cross_entropy,
    compute_metrics
)
import jax.numpy as jnp
import jax
from flax import nnx


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
    module = nnx.merge(state.graphdef, params, state.rest)
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


def eval_step(
  params: nnx.State,
  batch,
  graphdef: nnx.GraphDef,
  label_smoothing=0.0,
):
  """Calculate evaluation metrics on a batch."""
  inputs = batch['inputs']
  weights = jnp.where(inputs > 0, 1.0, 0.0)
  module = nnx.merge(graphdef, params)
  module.set_attributes(deterministic=True, decode=False)
  logits = module(inputs)

  return compute_metrics(logits, inputs, weights, label_smoothing)