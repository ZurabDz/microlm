import optax
from flax.training import train_state
from flax import nnx
import jax
from flax.training import common_utils
from flax import linen as nn
import numpy as np
import jax.numpy as jnp


class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef
  keys: nnx.RngKey
  rest: nnx.State


def rsqrt_schedule(
  init_value: float,
  shift: int = 0,
):
  """Applies a reverse square-root schedule.

  The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

  Args:
    init_value: Base learning rate (before applying the rsqrt schedule).
    shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
      schedule makes it less steep in the beginning (close to 0).

  Returns:
    A schedule that applies the reverse square root.
  """

  def schedule(count):
    return init_value * (count + shift) ** -0.5 * shift**0.5

  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules(
    [
      optax.linear_schedule(
        init_value=0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
      ),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
    ],
    boundaries=[warmup_steps],
  )

def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x


def compute_weighted_cross_entropy(
  logits, targets, weights=None, label_smoothing=0.0
):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
      'Incorrect shapes. Got shape %s logits and %s targets'
      % (str(logits.shape), str(targets.shape))
    )
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
    confidence * jnp.log(confidence)
    + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
  )
  soft_targets = common_utils.onehot(
    targets, vocab_size, on_value=confidence, off_value=low_confidence
  )

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
      'Incorrect shapes. Got shape %s logits and %s targets'
      % (str(logits.shape), str(targets.shape))
    )
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(
    logits, labels, weights, label_smoothing
  )
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
    'loss': loss,
    'accuracy': acc,
    'denominator': weight_sum,
  }
  return metrics


def setup_initial_state(
  constructor,
  tx,
  config,
  rng: jax.Array,
  mesh: jax.sharding.Mesh,
) -> tuple[TrainState, TrainState]:
  with mesh:
    model = constructor(config, rng)
    graphdef, params, keys, rest = nnx.split(model, nnx.Param, nnx.RngKey, ...)
    state = TrainState.create(
      apply_fn=graphdef.apply, params=params, tx=tx, graphdef=graphdef, keys=keys, rest=rest
    )
    state = jax.tree.map(_to_array, state)
    state_spec = nnx.get_partition_spec(state)
    state = jax.lax.with_sharding_constraint(state, state_spec)

  state_sharding = nnx.get_named_sharding(state, mesh)
  return state, state_sharding