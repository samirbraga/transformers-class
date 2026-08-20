import argparse
import json
import os
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import optax
from flax import nnx

from .checkpoint import load_parameters
from .generation import generate_with_top_k_sampling
from .interfaces import LanguageModel
from .tiny_shakespeare_data import load_tiny_shakespeare_corpus
from .mlp import MLPLanguageModel
from .train import train
from .transformer import TransformerLanguageModel
from .text_metrics import (
    TextMetrics,
    build_word_vocabulary,
    evaluate_generated_text,
    highlight_words_outside_vocabulary,
)
from .typings import TokenCorpus


CONTEXT_LENGTH: int = 64
BATCH_SIZE: int = 512
TRAINING_STEPS: int = 30_000
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-3
DROPOUT_RATE: float = 0.1
EVAL_INTERVAL: int = 500
GENERATION_TEMPERATURE: float = 0.7
GENERATION_TOP_K: int = 12
GENERATION_LENGTH: int = 500
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_MIN_DELTA: float = 1e-3
VALIDATION_BATCHES: int = 10

OUTPUT_DIR: Path = Path("output/tiny_shakespeare")
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
MLP_CHECKPOINT: Path = CHECKPOINT_DIR / "mlp.msgpack"
TRANSFORMER_CHECKPOINT: Path = CHECKPOINT_DIR / "transformer.msgpack"
LOSS_PLOT_FILE: Path = OUTPUT_DIR / "loss_comparison.png"
MLP_GENERATED_FILE: Path = OUTPUT_DIR / "mlp_generated.txt"
TRANSFORMER_GENERATED_FILE: Path = OUTPUT_DIR / "transformer_generated.txt"
METRICS_FILE: Path = OUTPUT_DIR / "generation_metrics.json"

MLP_EMBEDDING_DIM: int = 32
MLP_HIDDEN_DIM: int = 256

TRANSFORMER_D_MODEL: int = 128
TRANSFORMER_NUM_HEADS: int = 4
TRANSFORMER_NUM_LAYERS: int = 2
TRANSFORMER_D_FF: int = 512


def parameter_count(model: LanguageModel) -> int:
    parameters = nnx.state(model, nnx.Param)
    return sum(array.size for array in jax.tree.leaves(parameters))


def train_model[ModelT: LanguageModel](
    name: str,
    model: ModelT,
    train_data: TokenCorpus,
    validation_data: TokenCorpus,
    checkpoint_path: Path,
) -> tuple[ModelT, list[float], list[float]]:
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        wrt=nnx.Param,
    )

    print(f"\n--- Training {name} ---")
    print(f"Parameters: {parameter_count(model):,}")

    return train(
        model,
        optimizer,
        TRAINING_STEPS,
        BATCH_SIZE,
        EVAL_INTERVAL,
        train_data,
        validation_data,
        checkpoint_path,
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
    mlp_training_loss: list[float],
    mlp_validation_loss: list[float],
    transformer_training_loss: list[float],
    transformer_validation_loss: list[float],
) -> None:
    mlp_steps = evaluation_steps(len(mlp_training_loss))
    transformer_steps = evaluation_steps(len(transformer_training_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(
        mlp_steps,
        mlp_training_loss,
        color="tab:blue",
        linestyle="--",
        label="MLP training",
    )
    plt.plot(
        mlp_steps,
        mlp_validation_loss,
        color="tab:blue",
        label="MLP validation",
    )
    plt.plot(
        transformer_steps,
        transformer_training_loss,
        color="tab:orange",
        linestyle="--",
        label="Transformer training",
    )
    plt.plot(
        transformer_steps,
        transformer_validation_loss,
        color="tab:orange",
        label="Transformer validation",
    )
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("MLP vs Transformer on Tiny Shakespeare")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE, dpi=150)
    plt.close()


def generate_and_save(
    name: str,
    model: LanguageModel,
    validation_data: TokenCorpus,
    id_to_char: dict[int, str],
    seed: int,
    output_file: Path,
    word_vocabulary: set[str],
) -> str:
    initial_tokens = validation_data[: model.context_length]
    generated_text = generate_with_top_k_sampling(
        model,
        initial_tokens,
        id_to_char,
        model.context_length,
        jax.random.key(seed),
        length=GENERATION_LENGTH,
        temperature=GENERATION_TEMPERATURE,
        top_k=GENERATION_TOP_K,
    )
    output_file.write_text(generated_text, encoding="utf-8")
    print(f"\n--- {name} generated text ---\n")
    print(highlight_words_outside_vocabulary(generated_text, word_vocabulary))
    print(f"\nGenerated text saved to {output_file}")
    prompt_length = len("".join(id_to_char[int(token)] for token in initial_tokens))
    return generated_text[prompt_length:]


def create_models(
    vocab_size: int,
) -> tuple[MLPLanguageModel, TransformerLanguageModel]:
    mlp = MLPLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=MLP_EMBEDDING_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        context_length=CONTEXT_LENGTH,
        dropout_rate=DROPOUT_RATE,
        rngs=nnx.Rngs(42),
    )
    transformer = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=TRANSFORMER_D_MODEL,
        num_heads=TRANSFORMER_NUM_HEADS,
        num_layers=TRANSFORMER_NUM_LAYERS,
        d_ff=TRANSFORMER_D_FF,
        dropout_rate=DROPOUT_RATE,
        rngs=nnx.Rngs(43),
    )
    return mlp, transformer


def print_metrics(metrics: dict[str, TextMetrics]) -> None:
    print("\n--- Generated text metrics (continuation only) ---\n")
    print(
        f"{'Model':<14} {'Words':>7} {'Corpus words':>13} "
        f"{'Corpus rate':>12} {'Unique rate':>12} {'Avg length':>11}"
    )
    for name, model_metrics in metrics.items():
        print(
            f"{name:<14} {model_metrics.word_count:>7} "
            f"{model_metrics.corpus_word_count:>13} "
            f"{model_metrics.corpus_word_rate:>11.1%} "
            f"{model_metrics.unique_word_rate:>11.1%} "
            f"{model_metrics.average_word_length:>11.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MLP and Transformer LMs")
    parser.add_argument(
        "--mode",
        choices=("train", "generate"),
        default="train",
        help="Train both models or generate from their stored best checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_data, validation_data, char_to_id, id_to_char = (
        load_tiny_shakespeare_corpus()
    )
    print(f"Vocabulary size: {len(char_to_id)}")
    print(f"JAX devices: {jax.devices()}")

    mlp, transformer = create_models(len(char_to_id))

    if args.mode == "train":
        mlp, mlp_training_loss, mlp_validation_loss = train_model(
            "MLP",
            mlp,
            train_data,
            validation_data,
            MLP_CHECKPOINT,
        )
        transformer, transformer_training_loss, transformer_validation_loss = (
            train_model(
                "Transformer",
                transformer,
                train_data,
                validation_data,
                TRANSFORMER_CHECKPOINT,
            )
        )
        plot_losses(
            mlp_training_loss,
            mlp_validation_loss,
            transformer_training_loss,
            transformer_validation_loss,
        )
        print(f"\nLoss plot saved to {LOSS_PLOT_FILE}")
    else:
        load_parameters(mlp, MLP_CHECKPOINT)
        load_parameters(transformer, TRANSFORMER_CHECKPOINT)
        print(f"Loaded MLP checkpoint from {MLP_CHECKPOINT}")
        print(f"Loaded Transformer checkpoint from {TRANSFORMER_CHECKPOINT}")

    reference_text = "".join(id_to_char[int(token)] for token in train_data)
    word_vocabulary = build_word_vocabulary(reference_text)

    mlp_text = generate_and_save(
        "MLP",
        mlp,
        validation_data,
        id_to_char,
        seed=args.seed,
        output_file=MLP_GENERATED_FILE,
        word_vocabulary=word_vocabulary,
    )
    transformer_text = generate_and_save(
        "Transformer",
        transformer,
        validation_data,
        id_to_char,
        seed=args.seed,
        output_file=TRANSFORMER_GENERATED_FILE,
        word_vocabulary=word_vocabulary,
    )

    metrics = {
        "MLP": evaluate_generated_text(mlp_text, word_vocabulary),
        "Transformer": evaluate_generated_text(transformer_text, word_vocabulary),
    }
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
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
