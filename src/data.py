from itertools import chain

def process_dataset(dataset, tokenizer, n_proc=8):
    def tokenize_and_pad(batch):
        return {'ids': tokenizer(batch['text'])['input_ids']}

    def group_texts(examples, block_size=128):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result


    mapped_dataset = dataset.map(tokenize_and_pad, batched=True, num_proc=n_proc, remove_columns=dataset.column_names)
    grouped = mapped_dataset.map(group_texts, batched=True, num_proc=n_proc)

    return grouped