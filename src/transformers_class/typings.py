import numpy as np
from jaxtyping import Array, Float, Int

type TokenBatch = Int[Array, "batch context"]
type TokenTargets = Int[Array, " batch"]
type TokenCorpus = Int[np.ndarray, " corpus"]
type Logits = Float[Array, "batch vocab"]
type SequenceActivations = Float[Array, "batch context features"]
type AttentionWeights = Float[Array, "batch heads query key"]
type Scalar = Float[Array, ""]
