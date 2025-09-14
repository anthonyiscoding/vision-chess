import lightning as L
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from vision.model.data import NpyDataset
from vision.model.tokenizer import special_tokens_to_embeddings
from vision.pgn_to_npy import PgnProcessor


def collate_fn(batch):
    """Collate function to handle padding of sequences."""
    input_ids, target_ids = zip(*batch)
    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=special_tokens_to_embeddings["<|pad|>"],
    )
    target_ids = pad_sequence(
        target_ids,
        batch_first=True,
        padding_value=special_tokens_to_embeddings["<|pad|>"],
    )

    return input_ids, target_ids


class ChessDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, max_seq_len: int = 110):
        """Initialize the Chess Data Module.

        Args:
            batch_size: Number of samples per batch
            max_seq_len: Maximum sequence length for input sequences
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pgn_processor = PgnProcessor()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            training_files = self.pgn_processor.list_npy_files("data/training")
            validation_files = self.pgn_processor.list_npy_files("data/validation")

            self.train_dataset = NpyDataset(
                training_files, max_seq_len=self.max_seq_len
            )
            self.val_dataset = NpyDataset(
                validation_files, max_seq_len=self.max_seq_len
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
