import pandas as pd
import numpy as np
import yaml

from torch.utils.data import DataLoader
import torch

from lightning.pytorch import Trainer

from copy import deepcopy

from mbm.processors import make_default_spec, RowTransformer
from mbm.ftt import FTT
from pitch_data import PitchResultDataset

# Setup config ------------------

print("Reading config")
with open("config/simple_config.yaml", "r") as f:
    config = yaml.load(f, yaml.CLoader)

col_spec = make_default_spec(
    numeric_cols=config["data_types"]["numeric"],
    categorical_cols=config["data_types"]["categorical"],
)

# Setup data -------------------

pitch_df = pd.read_parquet("data/test_data.parquet")

# Imagine if pandas fillna() actually worked
for col in config["data_types"]["numeric"]:
    pitch_df[col] = pitch_df[col].astype(float)

pitch_df["result"] = pitch_df.description.map(config["result_map"])
result_cardinality = len(pitch_df["result"].unique())

train_df = pitch_df[pitch_df.game_date < "2022-09-01"]
validation_df = pitch_df[pitch_df.game_date >= "2022-09-01"]

train_row_trans = RowTransformer(col_spec)
train_row_trans.fit_to_data(train_df)

# This gross thing is to manage the mode of the row transformer's transformers
validation_row_trans = deepcopy(train_row_trans)
validation_row_trans.set_transformer_modes("eval")

train_ds = PitchResultDataset(train_df, train_row_trans)
train_dl = DataLoader(
    train_ds,
    batch_size=config["training_params"]["batch_size"],
    num_workers=config["training_params"]["num_workers"],
)
validation_ds = PitchResultDataset(validation_df, validation_row_trans)
validation_dl = DataLoader(
    validation_ds,
    batch_size=config["training_params"]["batch_size"],
    num_workers=config["training_params"]["num_workers"],
)

print(
    f"Loaded data with {len(train_ds)} training observations and {len(validation_ds)} validation observations."
)

# Setup model -------------------

net = FTT(
    row_transformer=train_row_trans,
    **config["net_params"],
    output_size=result_cardinality,
    optim_lr=config["training_params"]["learning_rate"],
    criterion=getattr(torch.nn, config["training_params"]["criterion"])(),
)

print(f"Created model with {sum(p.numel() for p in net.parameters())} weights.")


# Train the model ---------------

if __name__ == "__main__":
    trainer = Trainer(
        max_epochs=config["training_params"]["max_epochs"],
        log_every_n_steps=config["training_params"]["log_every_n"],
    )
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
