import torch
from torch.utils.data import Dataset

from mbm.processors import RobustIndexer


class PitchResultDataset(Dataset):
    RESULT_TYPES = [
        "swinging_strike",
        "called_strike",
        "foul",
        "ball",
        "hit_by_pitch",
        "in_play",
        "pitchout",
    ]

    def __init__(self, pitch_df, row_transformer):
        super().__init__()

        self.pitch_df = pitch_df
        self.row_transformer = row_transformer

        # We're re-using the RobustIndexer, but the Robust part isn't relevant
        # for targets.
        self.output_indexer = RobustIndexer(self.RESULT_TYPES, p_mask=0.0)
        self.output_indexer.set_mode("eval")

    def __getitem__(self, idx):
        row = self.pitch_df.iloc[idx]
        x = self.row_transformer.transform(row)
        y = self.output_indexer.transform(row["result"])

        return x, y

    def __len__(self):
        return len(self.pitch_df)
