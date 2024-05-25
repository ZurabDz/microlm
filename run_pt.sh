python train_mlm_torch.py \
    --model_type="albert" \
    --config_overrides="max_position_embeddings=32,hidden_size=768,intermediate_size=3072,num_attention_heads=12,num_hidden_layers=12" \
    --tokenizer_name="ZurabDz/culturax_bpe_32768" \
    --token="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" \
    --dataset_name="./tokenized_dataset" \
    --output_dir="./output2" \
    --mlm_probability 0.15 \
    --overwrite_output_dir true \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --warmup_ratio 0.1 \
    --learning_rate="9e-4" \
    --logging_steps 20 \
    --save_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed 42 \
    --fp16 true \
    --push_to_hub true \
    --report_to tensorboard \
    --hub_token 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 \
    --dataloader_prefetch_factor 8 \
    --dataloader_num_workers 4 \
    --max_eval_samples 100 