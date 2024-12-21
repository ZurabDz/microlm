### Grand plan: I want to have model in training on kaggle/colab tpus or somewhere. I want model to be generative for georgian language. Like GPTs but mine write on nnx library

### Couple of notes to take into account:
* my resources are restricted, I cant spin up gpus, more ram or more disk space.

#### Tokenizers
* I have trained BPE tokenizer from sentencepiece library, the problem is large the data more time it takes cause it has [not implemented parallel training for BPE](https://github.com/google/sentencepiece/issues/941). It does have num_threads for unigrams, but not for BPE.

* Training the tokenizer model
    * Currently will train model using [huggingface tokenizers library](https://github.com/huggingface/tokenizers).
    * What about beginning of sentence or end of sentence, other special tokens?
    * What about numbers and other non speaking characters?
    * Do we need BPE or others can crama more charecters per token?
* Processing the data
    * I am thinking using zstandard for saving jsonl data, which will be used for pretraining model as well as tokenizer
    * For starters I am gonna grab english data using huggingface dataset

#### Data processing
* I am thinking either using grain or huggingface dataset
    * I want to test out Grain for deterministic behaviour
    * I want to use huggingface cause its very popular and supports so many things out of the box

#### Modeling
* Probably will be based on [flax examples of nnx](https://github.com/google/flax/tree/main/examples/lm1b_nnx).
* I want minimal runable example even if it does not train well yet
* I want smaller size model < 1B parameters