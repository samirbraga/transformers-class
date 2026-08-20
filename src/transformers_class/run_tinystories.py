import argparse
import json
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import matplotlib.pyplot as plt
import optax
from flax import nnx
from tokenizers import Tokenizer

from .checkpoint import load_parameters
from .generation import generate_token_ids_with_top_k_sampling
from .interfaces import LanguageModel
from .mlp import MLPLanguageModel
from .text_metrics import (
    TextMetrics,
    build_word_vocabulary,
    evaluate_generated_text,
    highlight_words_outside_vocabulary,
)
from .tinystories_data import prepare_tinystories
from .train import train
from .transformer import TransformerLanguageModel
from .typings import TokenCorpus


TRAIN_STORIES: int = 20_000
VALIDATION_STORIES: int = 2_000
VOCAB_SIZE: int = 2_048
CONTEXT_LENGTH: int = 128
BATCH_SIZE: int = 64
TRAINING_STEPS: int = 30_000
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-3
DROPOUT_RATE: float = 0.1
EVAL_INTERVAL: int = 500
VALIDATION_BATCHES: int = 5
EARLY_STOPPING_PATIENCE: int = 8
EARLY_STOPPING_MIN_DELTA: float = 1e-3

MLP_EMBEDDING_DIM: int = 64
MLP_HIDDEN_DIM: int = 512
TRANSFORMER_D_MODEL: int = 192
TRANSFORMER_NUM_HEADS: int = 6
TRANSFORMER_NUM_LAYERS: int = 4
TRANSFORMER_D_FF: int = 768

MODEL_NAMES: dict[str, str] = {
    "mlp": "MLP",
    "transformer": "Transformer",
}

GENERATION_LENGTH: int = 256
GENERATION_TEMPERATURE: float = 0.8
GENERATION_TOP_K: int = 40

OUTPUT_DIR: Path = Path("output/tinystories")
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
LOSS_PLOT_FILE: Path = OUTPUT_DIR / "loss_comparison.png"
METRICS_FILE: Path = OUTPUT_DIR / "generation_metrics.json"


def parameter_count(model: LanguageModel) -> int:
    return sum(
        array.size for array in jax.tree.leaves(nnx.state(model, nnx.Param))
    )


def create_models(
    vocab_size: int,
) -> dict[str, MLPLanguageModel | TransformerLanguageModel]:
    return {
        "MLP": MLPLanguageModel(
            vocab_size,
            MLP_EMBEDDING_DIM,
            MLP_HIDDEN_DIM,
            CONTEXT_LENGTH,
            DROPOUT_RATE,
            rngs=nnx.Rngs(42),
        ),
        "Transformer": TransformerLanguageModel(
            vocab_size,
            CONTEXT_LENGTH,
            TRANSFORMER_D_MODEL,
            TRANSFORMER_NUM_HEADS,
            TRANSFORMER_NUM_LAYERS,
            TRANSFORMER_D_FF,
            DROPOUT_RATE,
            rngs=nnx.Rngs(43),
        ),
    }


def checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name.lower()}.msgpack"


def train_model[ModelT: LanguageModel](
    name: str,
    model: ModelT,
    train_data: TokenCorpus,
    validation_data: TokenCorpus,
) -> tuple[ModelT, list[float], list[float]]:
    print(f"\n--- Training TinyStories {name} ---")
    print(f"Parameters: {parameter_count(model):,}")
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        wrt=nnx.Param,
    )
    return train(
        model,
        optimizer,
        TRAINING_STEPS,
        BATCH_SIZE,
        EVAL_INTERVAL,
        train_data,
        validation_data,
        checkpoint_path(name),
        EARLY_STOPPING_PATIENCE,
        EARLY_STOPPING_MIN_DELTA,
        VALIDATION_BATCHES,
    )


def evaluation_steps(history_length: int) -> list[int]:
    steps = [
        step
        for step in range(1, TRAINING_STEPS + 1)
        if step == 1 or step % EVAL_INTERVAL == 0
    ]
    return steps[:history_length]


def plot_losses(
    histories: dict[str, tuple[list[float], list[float]]],
) -> None:
    colors = {"MLP": "tab:blue", "Transformer": "tab:orange"}
    plt.figure(figsize=(10, 6))
    for name, (training_loss, validation_loss) in histories.items():
        steps = evaluation_steps(len(training_loss))
        plt.plot(
            steps,
            training_loss,
            color=colors[name],
            linestyle="--",
            label=f"{name} training",
        )
        plt.plot(
            steps,
            validation_loss,
            color=colors[name],
            label=f"{name} validation",
        )
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("MLP vs Transformer on TinyStories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE, dpi=150)
    plt.close()


def generate_and_save(
    name: str,
    model: LanguageModel,
    validation_data: TokenCorpus,
    tokenizer: Tokenizer,
    word_vocabulary: set[str],
    seed: int,
) -> str:
    initial_tokens = validation_data[: model.context_length]
    generated_ids = generate_token_ids_with_top_k_sampling(
        model,
        initial_tokens,
        model.context_length,
        jax.random.key(seed),
        length=GENERATION_LENGTH,
        temperature=GENERATION_TEMPERATURE,
        top_k=GENERATION_TOP_K,
    )
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    continuation = tokenizer.decode(
        generated_ids[len(initial_tokens) :],
        skip_special_tokens=True,
    )
    output_file = OUTPUT_DIR / f"{name.lower()}_generated.txt"
    output_file.write_text(full_text, encoding="utf-8")
    print(f"\n--- TinyStories {name} generated text ---\n")
    print(highlight_words_outside_vocabulary(full_text, word_vocabulary))
    print(f"\nGenerated text saved to {output_file}")
    return continuation


def print_metrics(metrics: dict[str, TextMetrics]) -> None:
    print("\n--- TinyStories generated text metrics ---\n")
    print(
        f"{'Model':<14} {'Words':>7} {'Corpus words':>13} "
        f"{'Corpus rate':>12} {'Unique rate':>12} {'Avg length':>11}"
    )
    for name, values in metrics.items():
        print(
            f"{name:<14} {values.word_count:>7} {values.corpus_word_count:>13} "
            f"{values.corpus_word_rate:>11.1%} {values.unique_word_rate:>11.1%} "
            f"{values.average_word_length:>11.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train small subword language models on TinyStories"
    )
    parser.add_argument("--mode", choices=("train", "generate"), default="train")
    parser.add_argument(
        "--model",
        choices=("both", "mlp", "transformer"),
        default="transformer",
        help="Transformer is the recommended TinyStories experiment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_data, validation_data, tokenizer, reference_text = prepare_tinystories(
        train_stories=TRAIN_STORIES,
        validation_stories=VALIDATION_STORIES,
        vocab_size=VOCAB_SIZE,
    )
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(validation_data):,}")
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size():,}")
    print(f"JAX devices: {jax.devices()}")

    requested_names = (
        list(MODEL_NAMES.values())
        if args.model == "both"
        else [MODEL_NAMES[args.model]]
    )
    all_models = create_models(tokenizer.get_vocab_size())
    models = {name: all_models[name] for name in requested_names}
    histories: dict[str, tuple[list[float], list[float]]] = {}

    if args.mode == "train":
        for name, model in models.items():
            model, training_loss, validation_loss = train_model(
                name,
                model,
                train_data,
                validation_data,
            )
            models[name] = model
            histories[name] = (training_loss, validation_loss)
        plot_losses(histories)
        print(f"\nLoss plot saved to {LOSS_PLOT_FILE}")
    else:
        for name, model in models.items():
            path = checkpoint_path(name)
            load_parameters(model, path)
            print(f"Loaded {name} checkpoint from {path}")

    word_vocabulary = build_word_vocabulary(reference_text)
    metrics: dict[str, TextMetrics] = {}
    for name, model in models.items():
        continuation = generate_and_save(
            name,
            model,
            validation_data,
            tokenizer,
            word_vocabulary,
            args.seed,
        )
        metrics[name] = evaluate_generated_text(continuation, word_vocabulary)

    print_metrics(metrics)
    METRICS_FILE.write_text(
        json.dumps(
            {name: values.to_dict() for name, values in metrics.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nMetrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    main()
