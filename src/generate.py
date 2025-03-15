def generate_prediction(
  *,
  jit_pred_step,
  graphdef: nnx.GraphDef[models.TransformerLM],
  params: nnx.State,
  tokenized_prompts,
  eos_id,
  inference_rng,
  decode_tokens,
  config: default.Config,
  model_config: models.TransformerConfig,
):
  """Generate text from the prompt."""
  n_devices = jax.local_device_count()

  logging.info('Generating text.')
  predictions = []
  # Use batch of prompts provided by user.
  for pred_batch in jnp.array_split(
    tokenized_prompts, int(np.ceil(len(tokenized_prompts) / n_devices))
  ):
    cur_pred_batch_size = pred_batch.shape[0]
    if cur_pred_batch_size % n_devices:
      padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
      pred_batch = jax.tree.map(
        lambda x: pad_examples(x, padded_size), pred_batch
      )  # pylint: disable=cell-var-from-loop
    pred_batch = common_utils.shard(pred_batch)
    inference_rng, sub_rng = random.split(inference_rng)
    inference_rngs = random.split(sub_rng, n_devices)

    predicted = jit_pred_step(
      pred_batch,
      params,
      inference_rngs,
      graphdef,
      eos_id,
      config.max_predict_length,
      model_config,
      config.sampling_temperature,
      config.sampling_top_k,
    )
    predicted = tohost(predicted)
    # Iterate through non-padding examples of batch.
    for s in predicted[:cur_pred_batch_size]:
      prediction = decode_tokens(s)
      logging.info('Sample: %s', str(prediction))
      predictions.append(prediction)

    # Save generated texts for tensorboard.
    exemplars = ''
    for prediction in predictions:
      exemplars += f'{prediction}\n\n'
  return exemplars