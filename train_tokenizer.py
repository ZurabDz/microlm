from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer(BPE(vocab=32_768))
tokenizer.pre_tokenizer = Whitespace()

special_tokens = dict(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

trainer = BpeTrainer(special_tokens=list(special_tokens.values()))
tokenizer.train(files=['data.txt'], trainer=trainer)

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=1024 * 4,
    **special_tokens,
)

wrapped_tokenizer.save_pretrained('bpe-tokenizer-ka')