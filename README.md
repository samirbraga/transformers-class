# MLPs and Transformers from Scratch with JAX

This is a didactic language-modeling project. Its main experiment trains two autoregressive models on the same character-level Tiny Shakespeare corpus: a multilayer perceptron (MLP) that flattens a fixed context, and a causal Transformer that uses self-attention. An optional TinyStories experiment adds a small byte-level BPE tokenizer for more coherent subword generation.

The objective is not state-of-the-art text generation. It is to expose the architectural differences while keeping the dataset, vocabulary, next-token task, optimizer, evaluation, and sampling procedures as comparable as possible.

The implementation uses [JAX](https://docs.jax.dev/en/latest/), [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html), [Optax](https://optax.readthedocs.io/en/latest/), and [jaxtyping](https://docs.kidger.site/jaxtyping/).

## Repository architecture

```text
.
├── datasets/tiny_shakespeare/tinyshakespeare.txt
├── output/
│   ├── tiny_shakespeare/
│   │   ├── checkpoints/{mlp,transformer}.msgpack
│   │   ├── generation_metrics.json
│   │   ├── loss_comparison.png
│   │   ├── mlp_generated.txt
│   │   └── transformer_generated.txt
│   └── tinystories/
├── src/transformers_class/
│   ├── checkpoint.py       # NNX parameter persistence
│   ├── generation.py       # multinomial and top-k decoding
│   ├── interfaces.py       # shared LanguageModel protocol
│   ├── mlp.py              # flattened-context MLP
│   ├── run_tiny_shakespeare.py # Tiny Shakespeare experiment
│   ├── run_tinystories.py  # TinyStories subword experiment
│   ├── text_metrics.py     # lexical metrics and terminal highlighting
│   ├── tinystories_data.py # streaming, caching, and byte-level BPE
│   ├── tiny_shakespeare_data.py # Shakespeare download and encoding
│   ├── train.py            # batches, loss, optimization, early stopping
│   ├── transformer.py      # attention and Transformer blocks
│   ├── typings.py          # jaxtyping aliases
│   └── utils.py            # download and corpus helpers
├── pyproject.toml
└── README.md
```

`run_tiny_shakespeare.py` and `run_tinystories.py` are the experiment orchestrators. They load their respective data, construct models, call the shared trainer, plot losses, generate samples, and compute metrics. Generated artifacts live in dataset-specific directories under `output/`; datasets and outputs are ignored by Git.

## Getting started

The project requires [Python 3.13 or newer](https://docs.python.org/3/) and uses [uv](https://docs.astral.sh/uv/) for project and dependency management.

```bash
uv sync
```

Train both models, select their best checkpoints, generate text, and compute metrics:

```bash
uv run -m src.transformers_class.run_tiny_shakespeare --mode train
```

Generate again from stored checkpoints without training:

```bash
uv run -m src.transformers_class.run_tiny_shakespeare --mode generate
```

Train the recommended Transformer experiment on a streamed TinyStories subset:

```bash
uv run -m src.transformers_class.run_tinystories \
    --mode train --model transformer
```

Use `--model both` to train the MLP and Transformer on exactly the same subword tokens. Generate later from stored TinyStories checkpoints with:

```bash
uv run -m src.transformers_class.run_tinystories \
    --mode generate --model transformer
```

Generation mode expects `output/tiny_shakespeare/checkpoints/mlp.msgpack` and `output/tiny_shakespeare/checkpoints/transformer.msgpack`. Run training first if they do not exist.

The JAX dependency uses the CUDA 12 extra. Each experiment runner prints the devices detected by JAX. Consult the official [JAX installation and accelerator support guide](https://docs.jax.dev/en/latest/installation.html) when configuring CUDA. On a machine without a compatible NVIDIA setup, JAX may use CPU or report a CUDA initialization error.

## Main concepts

### Tokens and embeddings

A token is the discrete unit processed by a language model. Here every character—including spaces, punctuation, and line breaks—is a token. The sorted vocabulary contains about 65 symbols. Character tokenization avoids an external tokenizer, but makes the model learn spelling one character at a time.

Token IDs are categorical, so an embedding table maps each ID to a learned vector:

$$
E: \{0,\ldots,V-1\}\rightarrow\mathbb{R}^d.
$$

Both models use token embeddings. The Transformer also adds learned positional embeddings because attention alone does not encode order.

### Autoregression

An autoregressive model predicts the next token from earlier tokens:

$$
p(x_1,\ldots,x_n)=\prod_{t=1}^{n}p(x_t\mid x_1,\ldots,x_{t-1}).
$$

During generation, a sampled character is appended to the sequence and becomes part of the next input. Generation is therefore sequential and early mistakes can propagate.

### MLP

An MLP alternates affine transformations and nonlinearities:

$$
y=xW+b.
$$

This project uses GELU activations. The MLP flattens all context embeddings before its first Dense layer. Each position consequently reaches a different slice of that layer, giving implicit but hardwired position information.

Its limitations motivate attention:

- first-layer size grows with context length;
- weights are tied to absolute positions;
- a different context length requires a different first layer;
- sequence mixing is fixed rather than content-dependent.

### Transformer and self-attention

Self-attention constructs queries, keys, and values:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

then computes:

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad Y=AV.
$$

`M` is a causal mask: positions above the diagonal receive negative infinity before softmax, preventing access to future tokens. Multi-head attention repeats this operation in several subspaces, allowing different heads to represent different relationships.

Each block uses pre-LayerNorm and residual connections:

$$
x\leftarrow x+\mathrm{Attention}(\mathrm{LN}(x)),
$$

$$
x\leftarrow x+\mathrm{MLP}(\mathrm{LN}(x)).
$$

Residual paths improve gradient flow; LayerNorm stabilizes activations.

### Regularization, early stopping, and checkpoints

Dropout randomly removes activations during training and is disabled during validation/generation. AdamW weight decay discourages excessively large weights.

Early stopping detects when validation no longer improves even though training loss may keep falling. Validation is averaged over several batches. When it improves by at least `EARLY_STOPPING_MIN_DELTA`, parameters are saved. After `EARLY_STOPPING_PATIENCE` unsuccessful evaluations, training stops and the best checkpoint is restored.

Checkpoints contain model parameters only. They support inference, but not exact training resumption because optimizer state and the training step are not stored.

## Dataset

The project downloads [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), a roughly 1 MB collection of plays commonly used for small language-model demonstrations.

The loader:

1. downloads and caches the text;
2. creates a sorted character vocabulary;
3. encodes every character as an integer;
4. assigns the first 90% to training;
5. assigns the last 10% to validation.

The contiguous split avoids leaking nearly identical overlapping windows across train and validation. Tiny Shakespeare provides recurring speakers, dialogue, punctuation, and line structure, but it is not representative of general English.

### TinyStories and subword tokenization

The optional experiment uses [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a synthetic collection of short stories with deliberately simple language. It was designed for studying coherent generation in small language models.

The full dataset repository is several gigabytes, so `tinystories_data.py` uses [Hugging Face Datasets streaming](https://huggingface.co/docs/datasets/stream) and stops after a configurable number of examples. Defaults are 20,000 training stories and 2,000 validation stories. Stories are cached as JSONL, so later executions do not stream them again.

A byte-level BPE tokenizer is trained from scratch using [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/). Byte Pair Encoding merges frequent adjacent units, producing representations between individual characters and complete words. Common words can become one token while rare words remain multiple subwords. This reduces malformed spelling and lets a fixed token context cover more text than the character model.

The tokenizer uses NFKC normalization, a byte-level pre-tokenizer/decoder, a vocabulary of 2,048, and `<|unk|>`, `<|beginofstory|>`, and `<|endofstory|>` special tokens. The official [`BpeTrainer` documentation](https://huggingface.co/docs/tokenizers/main/api/trainers#tokenizers.trainers.BpeTrainer) describes its vocabulary and frequency settings.

Data is cached under a configuration-specific path:

```text
datasets/tinystories/train_20000_validation_2000_vocab_2048/
├── train.jsonl
├── validation.jsonl
├── tokenizer.json
├── train_tokens.npy
└── validation_tokens.npy
```

TinyStories results are isolated in `output/tinystories/`, with their own checkpoints, plot, generated texts, and metrics. The runner defaults to `--model transformer`; use `--model both` when direct architectural comparison matters more than training time.

## Models

### MLP architecture

```text
IDs [B,T]
  → Embed [B,T,E]
  → Flatten [B,T×E]
  → Dense + GELU + Dropout
  → Dense + GELU + Dropout
  → Dense [B,V]
```

Defaults:

```python
CONTEXT_LENGTH = 64
MLP_EMBEDDING_DIM = 32
MLP_HIDDEN_DIM = 256
DROPOUT_RATE = 0.1
```

The first Dense matrix has approximately `(64 × 32) × 256` weights. Doubling context while preserving other dimensions approximately doubles this layer.

### Transformer architecture

```text
IDs [B,T]
  → token embedding + learned position embedding [B,T,D]
  → TransformerBlock × L
  → final LayerNorm
  → select last position
  → Dense [B,V]
```

Each block contains causal multi-head attention and a `D → d_ff → D` GELU feed-forward network, both with pre-normalization, dropout, and residual connections.

Defaults:

```python
CONTEXT_LENGTH = 64
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_D_FF = 512
DROPOUT_RATE = 0.1
```

`TransformerLanguageModel.encode` also returns attention maps shaped `[batch, heads, query, key]` for visualization.

The current Transformer returns logits only for the last position. This deliberately matches the MLP objective—one next-character label per context—but underuses the Transformer. Standard Transformer training predicts shifted targets at every position.

Fewer parameters do not necessarily mean faster training. Attention creates `[B,H,T,T]` activations and performs projections, masking, softmax, value aggregation, normalization, and feed-forward computation at every position. The MLP has a large weight matrix but uses fewer, highly optimized matrix multiplications.

## Training

### Batch sampling

Each example samples a random corpus offset:

```python
inputs = data[start : start + context_length]
target = data[start + context_length]
```

Shapes are `[batch, context]` and `[batch]`. Both models restart the sampler with the same seed, so they receive the same context sequence. Validation uses a separate RNG and does not alter subsequent training batches.

### Loss and perplexity

The loss is mean next-character cross-entropy:

$$
\mathcal{L}=-\frac1B\sum_i\log p(y_i\mid x_i).
$$

`optax.softmax_cross_entropy_with_integer_labels` uses integer targets without one-hot allocation. Reported perplexity is:

$$
\mathrm{PPL}=e^{\mathcal{L}}.
$$

Perplexities are comparable here because both models use the same tokenization and validation data.

### Optimizer and configuration

Both models use:

```python
optax.adamw(learning_rate=3e-4, weight_decay=1e-3)
```

The default character experiment in `run_tiny_shakespeare.py` is:

```python
BATCH_SIZE = 512
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 500
VALIDATION_BATCHES = 10
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-3
```

Validation loss is the average of ten sampled batches. A batch of 512 may be efficient for the MLP but not optimal for Transformer throughput; reduce it to 128 or 256 if needed.

The comparison plot is saved as `output/tiny_shakespeare/loss_comparison.png`.

## Sampling strategies

Both strategies are autoregressive and use explicit JAX PRNG keys.

### Multinomial sampling

`generate_with_multinomial_sampling` samples from the entire vocabulary after temperature scaling:

$$
p_i=\mathrm{softmax}(z_i/\tau).
$$

Low temperature is conservative and may repeat; high temperature is diverse and makes malformed words more likely.

### Top-k sampling

`generate_with_top_k_sampling` retains only the `k` highest-logit characters and samples among them. The comparison uses:

```python
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_K = 12
GENERATION_LENGTH = 500
```

The same settings and seed are used for both models. Text is saved to `output/tiny_shakespeare/mlp_generated.txt` and `output/tiny_shakespeare/transformer_generated.txt`. Terminal output colors words absent from the training vocabulary red; files contain no ANSI codes.

## Generated-text metrics

Metrics exclude the shared prompt and evaluate only the continuation:

- `word_count`: word-token occurrences;
- `corpus_word_count`: occurrences found in the training vocabulary;
- `corpus_word_rate`: recognized occurrences divided by total words;
- `unique_word_count` and `unique_word_rate`;
- `average_word_length`.

For Tiny Shakespeare they are saved to `output/tiny_shakespeare/generation_metrics.json`; TinyStories uses `output/tinystories/generation_metrics.json`.

Corpus-word rate is lexical plausibility, not grammar or coherence. Copying common words can score highly, while valid modern words absent from Shakespeare count as unknown. Use it alongside validation loss and qualitative inspection.

## Implementation details

### [JAX](https://docs.jax.dev/en/latest/)

[JAX's quickstart](https://docs.jax.dev/en/latest/quickstart.html) introduces its NumPy-like operations (`jax.numpy`), automatic differentiation, XLA compilation, and accelerator execution. The core update in this project is:

```python
loss, gradients = nnx.value_and_grad(loss_fn)(model, inputs, targets)
optimizer.update(model, gradients)
```

`@nnx.jit` compiles training and evaluation. The first call includes compilation overhead; later calls reuse the compiled program. Changing array shapes can trigger recompilation.

Randomness is explicit and reproducible:

```python
key, sample_key = jax.random.split(key)
```

### [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html)

[Flax](https://flax.readthedocs.io/en/latest/) supplies the NNX API for stateful `nnx.Module` objects and layers including `Linear`, `Embed`, `LayerNorm`, and `Dropout`. Models are initialized with `nnx.Rngs`; parameters are selected with `nnx.state(model, nnx.Param)`. The Transformer uses `nnx.List` so nested blocks are registered correctly. See the official [NNX API reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/index.html) for the complete module and transformation APIs.

Dropout mode is explicit:

```python
model(inputs, deterministic=False)  # training
model(inputs, deterministic=True)   # validation/generation
```

For checkpoints, NNX parameter states are converted to pure dictionaries, serialized with Flax MessagePack support, reconstructed against a fresh template, and applied with `nnx.update`.

`LanguageModel` is a structural Python `Protocol` requiring `context_length` and `__call__(tokens, deterministic=...)`. This lets the same trainer and generator work with both architectures without inheritance coupling.

### [jaxtyping](https://docs.kidger.site/jaxtyping/)

[jaxtyping's documentation](https://docs.kidger.site/jaxtyping/api/array/) describes the array annotations used here to document dtype and named shapes:

```python
type TokenBatch = Int[Array, "batch context"]
type Logits = Float[Array, "batch vocab"]
type AttentionWeights = Float[Array, "batch heads query key"]
type Scalar = Float[Array, ""]
```

These annotations clarify tensor contracts. Runtime shape checking is not enabled in this project.

### [Optax](https://optax.readthedocs.io/en/latest/)

[Optax](https://optax.readthedocs.io/en/latest/) provides AdamW and numerically stable integer-label cross-entropy. Its official [getting-started guide](https://optax.readthedocs.io/en/latest/getting_started.html) explains gradient transformations and optimizer state. NNX connects an Optax transformation to model parameters with:

```python
optimizer = nnx.Optimizer(
    model,
    optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    wrt=nnx.Param,
)
```

## Interpreting and extending the experiment

Compare validation loss/perplexity, best step, parameter count, runtime, lexical metrics, multiple generated seeds, and attention maps. One sample alone is weak evidence, and fewer parameters do not imply less computation.

Useful extensions include:

- full-sequence shifted-target training for the Transformer;
- learning-rate warmup and decay;
- saving optimizer state for resumable training;
- multiple generation seeds with aggregated metrics;
- associative-recall data for a direct long-range attention demonstration;
- TinyStories with a small subword tokenizer for more coherent language.
