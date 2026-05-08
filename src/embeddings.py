"""
Embedding extraction utilities for BENADV-style multilingual experiments.

The main purpose of this module is to extract token-level contextual hidden
states from HuggingFace transformer models. These hidden states are later
flattened and passed to Benford-law feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class EmbeddingResult:
    """Token-level embedding result for one text."""

    model_name: str
    text: str
    hidden_states: np.ndarray
    n_tokens: int
    hidden_size: int

    @property
    def n_numbers(self) -> int:
        return int(self.hidden_states.size)

    def flatten(self) -> np.ndarray:
        return self.hidden_states.reshape(-1)


class TransformerEmbeddingExtractor:
    """
    Extract final-layer contextual token embeddings from a transformer model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``xlm-roberta-base``.
    device:
        ``cuda``, ``cpu``, or ``None``. If ``None``, CUDA is used when available.
    max_length:
        Maximum tokenized sequence length. Long texts are truncated.
    include_special_tokens:
        If False, special tokens such as CLS/SEP/BOS/EOS are removed before
        returning the hidden-state matrix.
    trust_remote_code:
        Passed to HuggingFace loaders. Keep False unless a specific model needs it.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        max_length: int = 512,
        include_special_tokens: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.include_special_tokens = include_special_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str) -> EmbeddingResult:
        """
        Encode one text and return final hidden states as a NumPy array.

        Returns shape ``(n_tokens, hidden_size)``.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

        special_tokens_mask = encoded.pop("special_tokens_mask")[0].cpu().numpy().astype(bool)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)

        if not self.include_special_tokens:
            hidden = hidden[~special_tokens_mask]

        return EmbeddingResult(
            model_name=self.model_name,
            text=text,
            hidden_states=hidden,
            n_tokens=int(hidden.shape[0]),
            hidden_size=int(hidden.shape[1]) if hidden.ndim == 2 else 0,
        )

    def encode_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """
        Encode several texts sequentially.

        This intentionally uses sequential encoding to keep implementation simple
        and to avoid padding artifacts in the first version of the experiment.
        """
        return [self.encode(text) for text in texts]

    def token_count(self, text: str) -> int:
        """Return token count under this tokenizer, optionally excluding special tokens."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

        if self.include_special_tokens:
            return int(len(encoded["input_ids"]))

        mask = np.asarray(encoded["special_tokens_mask"], dtype=bool)
        return int((~mask).sum())


def extract_embedding_matrix(
    text: str,
    model_name: str = "xlm-roberta-base",
    *,
    device: Optional[str] = None,
    max_length: int = 512,
    include_special_tokens: bool = False,
    trust_remote_code: bool = False,
) -> np.ndarray:
    """
    Convenience function for one-off embedding extraction.

    For large experiments, instantiate ``TransformerEmbeddingExtractor`` once
    and reuse it instead of calling this repeatedly.
    """
    extractor = TransformerEmbeddingExtractor(
        model_name,
        device=device,
        max_length=max_length,
        include_special_tokens=include_special_tokens,
        trust_remote_code=trust_remote_code,
    )
    return extractor.encode(text).hidden_states


if __name__ == "__main__":
    extractor = TransformerEmbeddingExtractor("xlm-roberta-base", device="cpu")
    result = extractor.encode("This is a short smoke test.")
    print(
        {
            "model": result.model_name,
            "shape": result.hidden_states.shape,
            "n_tokens": result.n_tokens,
            "n_numbers": result.n_numbers,
        }
    )
