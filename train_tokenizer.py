import zstandard as zstd
from tokenizers import SentencePieceBPETokenizer
import json
from datasets import load_dataset
from tqdm import tqdm


# Loading tiny stories dataset
ds = load_dataset("roneneldan/TinyStories")


# # Writing train data to a compressed JSONL file
# with zstd.open("tiny_stories_train.jsonl.zst", "w") as f:
#     for record in tqdm(ds['train']):
#         f.write(json.dumps(record) + "\n")


def iterate_over_zst_data(path):
    with zstd.open(path, "r") as f:
        for line in f:
            record = json.loads(line)['text']
            yield record

# def iterate_over_zst_data(path):
#     dctx = zstd.ZstdDecompressor()
#     with open(path, 'rb') as compressed:
#         reader = dctx.stream_reader(compressed)
#         buffer = ''
#         while True:
#             chunk = reader.read(2**20)  # 1MB chunks
#             if not chunk:
#                 break
#             buffer += chunk.decode('utf-8')
#             lines = buffer.split('\n')

#             data = []
#             for line in lines[:-1]:  # Process all lines except the last one which might be incomplete
#                 if line:  # Skip empty lines if any
#                     try:
#                         record = json.loads(line)
#                         data.append(record.get('text', None))
#                     except json.JSONDecodeError:
#                         print(f"Error decoding line: {line}")

#             yield data
#             buffer = lines[-1]  # Keep the last (possibly incomplete) line for the next iteration

tokenizer = SentencePieceBPETokenizer()
train_data_iterator = iterate_over_zst_data('tiny_stories_train.jsonl.zst')

tokenizer.train_from_iterator(
    train_data_iterator,
    vocab_size=30000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<|begin_of_text|>", "<|end_of_text|>"],
)

tokenizer.save_model('./')
