import torch
from torch.utils.data import Dataset

from mbm.processors import RobustIndexer


class PitchResultDataset(Dataset):
    def __init__(self, pitch_df, x_transformer, y_transformer):
        super().__init__()

        self.x_transformer = x_transformer
        self.y_transformer = y_transformer

        self.x_df = x_transformer.transform(pitch_df)
        self.y_df = y_transformer.transform(pitch_df)

    def __getitem__(self, idx):
        x_row = self.x_df.iloc[[idx]]
        y_row = self.y_df.iloc[[idx]]
        x = {
            col: torch.tensor(x_row[col].item()).unsqueeze(-1) for col in x_row.columns
        }
        y = torch.tensor(y_row["result"].item())

        return x, y

    def __len__(self):
        return len(self.x_df)
