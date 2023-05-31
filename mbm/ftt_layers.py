import torch
from torch import nn

import math

from .processors import RobustIndexer, Normalizer


class LearnedImputer(nn.Module):
    def __init__(self, initial_value=0.0):
        super().__init__()
        self.imputed_value = nn.Parameter(torch.tensor(initial_value))

    def forward(self, x):
        x[x.isnan()] = self.imputed_value
        return x


class NumericTokenizer(nn.Sequential):
    def __init__(self, d_model):
        super().__init__()

        self.append(LearnedImputer())
        self.append(nn.Linear(1, d_model))
        self.append(nn.LayerNorm(d_model))
        self.append(nn.ReLU())


class CategoricalTokenizer(nn.Sequential):
    def __init__(self, n_embeddings, d_model):
        super().__init__()

        # Expand embedding size to always be divisible by 8.
        # This wastes embeddings but may be faster.
        n_embeddings = math.ceil(n_embeddings / 8) * 8
        self.append(nn.Embedding(n_embeddings, d_model))
        self.append(nn.LayerNorm(d_model))
        self.append(nn.ReLU())

    def forward(self, x):
        # Oh no
        return super().forward(x).squeeze(1)


class FTTInputLayer(nn.Module):
    """This is the simplest possible version of this. Notably:
    - The classification token is obligatory.
    - There's no column encodings.
    - No real thought has been given to pre-training, yet."""

    def __init__(self, row_transformer, *args, **kwargs):
        super().__init__()

        # We don't need the actual transforms of the row_transformer, but we do
        # want the type and cardinality information that it contains.
        self.col_spec = row_transformer.column_transformers

        # In here so you can easily inherit this class and overwrite just this method.
        self.tokenizers = self.make_tokenizers(args, kwargs)

        # I'm going to leave out the column encodings for the moment because I'm curious.

        self.cls_embedding = nn.Parameter(
            torch.normal(
                torch.zeros([1, 1, kwargs["d_model"]]),
                1.0,
            )
        )

    def make_tokenizers(self, args, kwargs):
        """This is the default one."""

        tokenizers = nn.ModuleDict()

        for col, processor in self.col_spec.items():
            if isinstance(processor, RobustIndexer):
                tokenizers[col] = CategoricalTokenizer(
                    n_embeddings=len(processor), d_model=kwargs["d_model"]
                )

            elif isinstance(processor, Normalizer):
                tokenizers[col] = NumericTokenizer(d_model=kwargs["d_model"])

            else:
                raise ValueError(
                    "The default tokenizer only supports 'RobustIndexer' and 'Normalizer' processors."
                )

        return tokenizers

    def forward(self, x):
        # n x e x s

        x = torch.stack(
            [
                tokenizer(x[col]).squeeze(-1)
                for col, tokenizer in self.tokenizers.items()
            ],
            dim=-1,
        )

        # permute tensor to n x s x e
        x = torch.permute(x, [0, 2, 1])

        # Add the CLS token.
        expanded_cls = self.cls_embedding.expand([x.shape[0], -1, -1])
        x = torch.cat([x, expanded_cls], dim=1)

        return x


class FTTOutputLayer(nn.Sequential):
    def __init__(self, d_model, output_size, n_layers=1):
        super().__init__()
        for _ in range(n_layers):
            self.append(nn.LayerNorm(d_model))
            self.append(nn.ReLU())
            self.append(nn.Linear(d_model, output_size))
