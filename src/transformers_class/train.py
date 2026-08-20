from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .checkpoint import load_parameters, save_parameters
from .interfaces import LanguageModel
from .typings import Scalar, TokenBatch, TokenCorpus, TokenTargets




def sample_batch(
    data: TokenCorpus,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
) -> tuple[TokenBatch, TokenTargets]:
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    inputs = np.stack([data[start : start + context_length] for start in starts])
    targets = data[starts + context_length]
    return jnp.asarray(inputs), jnp.asarray(targets)


def loss_fn(
    model: LanguageModel,
    inputs: TokenBatch,
    targets: TokenTargets,
) -> Scalar:
    logits = model(inputs, deterministic=False)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


@nnx.jit
def train_step(
    model: LanguageModel,
    optimizer: nnx.Optimizer,
    inputs: TokenBatch,
    targets: TokenTargets,
) -> Scalar:
    loss, gradients = nnx.value_and_grad(loss_fn)(model, inputs, targets)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def evaluate(
    model: LanguageModel,
    inputs: TokenBatch,
    targets: TokenTargets,
) -> Scalar:
    logits = model(inputs, deterministic=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def train[ModelT: LanguageModel](
    model: ModelT,
    optimizer: nnx.Optimizer,
    training_steps: int,
    batch_size: int,
    eval_interval: int,
    train_data: TokenCorpus,
    validation_data: TokenCorpus,
    checkpoint_path: Path,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    validation_batches: int = 10,
) -> tuple[ModelT, list[float], list[float]]:
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if validation_batches < 1:
        raise ValueError("validation_batches must be positive")

    training_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    best_validation_loss = float("inf")
    best_step = 0
    evaluations_without_improvement = 0

    training_rng = np.random.default_rng(42)
    validation_rng = np.random.default_rng(43)
    for step in range(1, training_steps + 1):
        inputs, targets = sample_batch(
            train_data,
            batch_size,
            model.context_length,
            training_rng,
        )
        training_loss = train_step(model, optimizer, inputs, targets)

        if step == 1 or step % eval_interval == 0:
            validation_losses: list[float] = []
            for _ in range(validation_batches):
                validation_inputs, validation_targets = sample_batch(
                    validation_data,
                    batch_size * 2,
                    model.context_length,
                    validation_rng,
                )
                validation_losses.append(
                    float(evaluate(model, validation_inputs, validation_targets))
                )
            validation_loss = float(np.mean(validation_losses))

            validation_loss_history.append(validation_loss)
            training_loss_history.append(float(training_loss))

            print(
                f"step={step:5d} "
                f"train_loss={float(training_loss):.4f} "
                f"validation_loss={validation_loss:.4f} "
                f"perplexity={float(jnp.exp(validation_loss)):.2f}"
            )

            if validation_loss < best_validation_loss - early_stopping_min_delta:
                best_validation_loss = validation_loss
                best_step = step
                evaluations_without_improvement = 0
                save_parameters(model, checkpoint_path)
                print(f"Saved new best checkpoint to {checkpoint_path}")
            else:
                evaluations_without_improvement += 1
                if evaluations_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping at step {step}; best validation loss "
                        f"was {best_validation_loss:.4f} at step {best_step}."
                    )
                    break

    load_parameters(model, checkpoint_path)
    print(f"Restored best checkpoint from step {best_step}.")
    return model, training_loss_history, validation_loss_history
