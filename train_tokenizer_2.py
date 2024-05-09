from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = SentencePieceBPETokenizer(unk_token='[UNK]')

special_tokens = dict(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

tokenizer.train('data/zdata.txt', special_tokens=list(special_tokens.values()),
                min_frequency=100, vocab_size=32_768, limit_alphabet=100)

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=1024 * 4,
    **special_tokens,
)

wrapped_tokenizer.save_pretrained('tok2')