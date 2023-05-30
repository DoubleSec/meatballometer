import torch
import numpy as np

from random import random


class RobustIndexer:
    """Creates consistent mappings from categorical values to integers. The 'Robust'
    part is a little more involved.

    During training (with mode set to 'train'), the indexer masks a percentage of
    actual values it gets, and returns an 'unknown' in their place. This allows the
    the network to learn an 'unknown' embedding even if there are no unknowns in the
    training data. Then during testing/inference, this behavior can be disabled by
    setting the mode to 'eval', which uses all available data."""

    def __init__(self, values, p_mask=0.01):
        self.map = {"<UNK>": torch.tensor(0, dtype=torch.long)}

        for value in values:
            if value not in self.map:
                self.map[value] = torch.tensor(len(self.map), dtype=torch.long)

        self.inverse_map = {v: k for k, v in self.map.items()}

        self.mode = "train"
        self.p_mask = p_mask

    def set_mode(self, mode):
        if mode not in ["train", "eval"]:
            raise ValueError("mode must either 'train' or 'eval'")

        else:
            self.mode = mode

    def transform(self, x):
        if self.mode == "train":
            return self.map[x] if random() > self.p_mask else self.map["<UNK>"]

        # If in eval mode
        else:
            return self.map.get(x, self.map["<UNK>"])

    def inverse_transform(self, x):
        return self.inverse_map["x"]

    @classmethod
    def from_data(cls, data, p_mask=0.01):
        values = np.unique(data)

        return cls(values, p_mask=p_mask)

    def __repr__(self):
        return f"RobustIndexer({[k for k in self.map.keys()]}, p_mask={self.p_mask})"

    def __len__(self):
        return len(self.map)


class Normalizer:
    """Applies a very simple transformation to numeric data, subtracting the mean
    and dividing by the standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        return torch.tensor((x - self.mean) / self.std, dtype=torch.float).unsqueeze(-1)

    def inverse_transform(self, x):
        return x * self.std + self.mean

    @classmethod
    def from_data(cls, data):
        mean = np.mean(data)
        std = np.std(data)

        return cls(mean, std)

    def __repr__(self):
        return f"Normalizer(mean={self.mean :.3f}, std={self.std :.3f})"


class RowTransformer:
    """This class collects a set of individual column transformers and can
    apply them to a 'row'. Typically that's a row of a pandas dataframe, but
    it should also work on a pyspark row and maybe a namedtuple.

    The normal way is to provide a column spec when the object is created
    (like that made by make_spec), and then call fit_to_data on some data.

    If you want to use your own transformers, they must implement transform and
    from_data methods, and optionally also __repr__, inverse_transform and
    set_mode (if you need it)."""

    def __init__(self, column_spec):
        self.column_spec = column_spec

    def fit_to_data(self, data):
        self.column_transformers = {
            col: processor_type.from_data(data[col])
            for col, processor_type in self.column_spec.items()
        }

    def transform(self, x):
        return {
            col: transformer.transform(x[col])
            for col, transformer in self.column_transformers.items()
        }

    def set_transformer_modes(self, mode):
        for transformer in self.column_transformers.values():
            if hasattr(transformer, "set_mode"):
                transformer.set_mode(mode)


def make_default_spec(numeric_cols, categorical_cols):
    """This is a convenience function that only works if you only have categorical
    and numeric features. If you have different kinds of transformations to do,
    you're on your own, although hopefully this is a useful pattern."""

    col_spec = {col: Normalizer for col in numeric_cols}
    # This is dictionary union
    col_spec |= {col: RobustIndexer for col in categorical_cols}

    return col_spec


if __name__ == "__main__":
    import pandas as pd
    import yaml

    print("Reading data")
    pitch_df = pd.read_parquet("data/test_data.parquet")

    with open("spec/data_spec.yaml", "r") as f:
        data_spec = yaml.load(f, yaml.CLoader)

    col_spec = make_default_spec(
        numeric_cols=data_spec["data_types"]["numeric"],
        categorical_cols=data_spec["data_types"]["categorical"],
    )

    row_trans = RowTransformer(col_spec)
    row_trans.fit_to_data(pitch_df)

    print(row_trans.column_transformers)

    print(row_trans.transform(pitch_df.iloc[0]))
