from datasets import load_dataset
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC


ds = load_dataset('DKYoon/SlimPajama-6B', num_proc=6)
tokenizer = Tokenizer(BPE(unk_token='<unk>'))  # do I need unk


tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

data = ds['test']

trainer = BpeTrainer(vocab_size=30_000, min_frequency=2, show_progress=True,
                     special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

tokenizer.train_from_iterator(
    (entry['text'] for entry in data.iter(batch_size=1000)), trainer, length=len(data)
)

tokenizer.save('fresh_tokenizer_bpe.json')