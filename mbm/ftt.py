import torch
from torch import nn

import lightning.pytorch as pl
from torchmetrics.classification import AUROC, Accuracy

from .ftt_layers import FTTInputLayer, FTTOutputLayer


class FTT(pl.LightningModule):
    def __init__(
        self,
        row_transformer,
        d_model,
        output_size,
        output_n_layers,
        transformer_n_layers,
        transformer_n_heads,
        transformer_dim_ff,
        optim_lr,
        criterion,
    ):
        super().__init__()

        self.input_layer = FTTInputLayer(row_transformer, d_model=d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_n_heads,
                dim_feedforward=transformer_dim_ff,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=transformer_n_layers,
        )
        self.output_layer = FTTOutputLayer(
            d_model=d_model, output_size=output_size, n_layers=output_n_layers
        )

        # Optimizer and loss function settings.
        self.optim_lr = optim_lr
        self.criterion = criterion

        # Logging
        self.train_auroc = AUROC("multiclass", num_classes=output_size)
        self.valid_auroc = AUROC("multiclass", num_classes=output_size)
        self.train_accuracy = Accuracy("multiclass", num_classes=output_size)
        self.valid_accuracy = Accuracy("multiclass", num_classes=output_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        # This gets the CLS token
        x = self.output_layer(x[:, -1, :])

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_auroc(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_auc": self.train_auroc,
                "train_accuracy": self.train_accuracy,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.valid_auroc(y_hat, y)
        self.valid_accuracy(y_hat, y)
        self.log_dict(
            {
                "valid_loss": loss,
                "valid_auc": self.valid_auroc,
                "valid_accuracy": self.valid_accuracy,
            }
        )

        return loss
