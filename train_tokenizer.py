from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

special_tokens = dict(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

trainer = BpeTrainer(special_tokens=list(special_tokens.values()), vocab_size=32_768)
tokenizer.train(files=['data/zdata.txt'], trainer=trainer)

tokenizer.save('tok/tokenizer')

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=1024 * 4,
    **special_tokens,
)

wrapped_tokenizer.save_pretrained('bpe-tokenizer-ka')