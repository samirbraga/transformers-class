from flax import nnx
from .typings import TokenBatch, Logits

class MLPLanguageModel(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        context_length: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.context_length = context_length
        self.embedding = nnx.Embed(vocab_size, embedding_dim, rngs=rngs)
        self.hidden_1 = nnx.Linear(
            context_length * embedding_dim,
            hidden_dim,
            rngs=rngs,
        )
        self.hidden_2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, tokens: TokenBatch, *, deterministic: bool) -> Logits:
        x = self.embedding(tokens)
        x = x.reshape((x.shape[0], -1))
        x = nnx.gelu(self.hidden_1(x))
        x = self.dropout(x, deterministic=deterministic)
        x = nnx.gelu(self.hidden_2(x))
        x = self.dropout(x, deterministic=deterministic)
        return self.output(x)
