import argparse
import logging
import sys
from multiprocessing import freeze_support
import torch
from torch.utils.data import DataLoader
from vision.model import transformer as t
from vision.model.data import NpyDataset
from vision.pgn_to_npy import list_npy_files
from vision.model.tokenizer import special_tokens_to_embeddings
from vision.train import train
from torch.nn.utils.rnn import pad_sequence


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


logger = logging.getLogger(__name__)


def collate_fn(batch):
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


def main(config):
    device = torch.device("mps")

    training_files = list_npy_files("data/training")
    validation_files = list_npy_files("data/validation")

    training_dataset = NpyDataset(training_files)
    validation_dataset = NpyDataset(validation_files)

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=2,
    )

    model = t.ChessModel(config)
    train(
        model=model,
        device=device,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-like model to play Chess")
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    if args.test:
        from vision.model.config.test import config
    else:
        from vision.model.config.default import config

    setup_logging()
    freeze_support()
    main(config=config)
