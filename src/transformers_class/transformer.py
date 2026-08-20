import math

import jax
import jax.numpy as jnp
from flax import nnx

from .typings import AttentionWeights, Logits, SequenceActivations, TokenBatch


class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.qkv_projection = nnx.Linear(d_model, 3 * d_model, rngs=rngs)
        self.output_projection = nnx.Linear(d_model, d_model, rngs=rngs)
        self.attention_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: SequenceActivations,
        *,
        deterministic: bool,
    ) -> tuple[SequenceActivations, AttentionWeights]:
        batch_size, context_length, d_model = x.shape

        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(
            batch_size,
            context_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        queries, keys, values = jnp.moveaxis(qkv, 2, 0)

        attention_logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            queries,
            keys,
        ) / math.sqrt(self.head_dim)

        causal_mask = jnp.tril(
            jnp.ones((context_length, context_length), dtype=jnp.bool_)
        )
        attention_logits = jnp.where(causal_mask, attention_logits, -jnp.inf)
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        attention_weights = self.attention_dropout(
            attention_weights,
            deterministic=deterministic,
        )

        attended_values = jnp.einsum(
            "bhqk,bkhd->bqhd",
            attention_weights,
            values,
        )
        attended_values = attended_values.reshape(
            batch_size,
            context_length,
            d_model,
        )
        output = self.output_projection(attended_values)
        output = self.output_dropout(output, deterministic=deterministic)
        return output, attention_weights


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.attention = CausalSelfAttention(
            d_model,
            num_heads,
            dropout_rate,
            rngs=rngs,
        )
        self.feed_forward_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.feed_forward_in = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.feed_forward_out = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.feed_forward_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: SequenceActivations,
        *,
        deterministic: bool,
    ) -> tuple[SequenceActivations, AttentionWeights]:
        attention_output, attention_weights = self.attention(
            self.attention_norm(x),
            deterministic=deterministic,
        )
        x = x + attention_output

        feed_forward = self.feed_forward_norm(x)
        feed_forward = nnx.gelu(self.feed_forward_in(feed_forward))
        feed_forward = self.feed_forward_out(feed_forward)
        feed_forward = self.feed_forward_dropout(
            feed_forward,
            deterministic=deterministic,
        )
        return x + feed_forward, attention_weights


class TransformerLanguageModel(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.context_length: int = context_length
        self.token_embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.position_embedding = nnx.Embed(context_length, d_model, rngs=rngs)
        self.embedding_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.blocks = nnx.List(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout_rate,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.output = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def encode(
        self,
        tokens: TokenBatch,
        *,
        deterministic: bool,
    ) -> tuple[SequenceActivations, list[AttentionWeights]]:
        sequence_length = tokens.shape[1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds maximum context "
                f"length {self.context_length}"
            )

        positions = jnp.arange(sequence_length)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.embedding_dropout(x, deterministic=deterministic)

        attention_maps: list[AttentionWeights] = []
        for block in self.blocks:
            x, attention_weights = block(x, deterministic=deterministic)
            attention_maps.append(attention_weights)

        return self.final_norm(x), attention_maps

    def __call__(
        self,
        tokens: TokenBatch,
        *,
        deterministic: bool,
    ) -> Logits:
        x, _ = self.encode(tokens, deterministic=deterministic)
        return self.output(x[:, -1, :])
