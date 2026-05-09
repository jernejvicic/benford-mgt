import sys
sys.path.append("src")
from embeddings import TransformerEmbeddingExtractor


extractor = TransformerEmbeddingExtractor(
    "xlm-roberta-base",
    device="cpu"
)

result = extractor.encode("Hello world!")

print(result.hidden_states.shape)

