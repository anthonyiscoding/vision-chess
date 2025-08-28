import logging
import sys
from multiprocessing import freeze_support
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from vision.model import transformer as t
from vision.model.datamodule import ChessDataModule
from vision.model.config import config
from vision.utils import get_device


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


logger = logging.getLogger(__name__)


def main(config):
    data_module = ChessDataModule(config)
    model = t.ChessModel(config)

    callbacks = []

    if config.save_model:
        # Lightning checkpoint callback - saves .ckpt files
        checkpoint_callback = ModelCheckpoint(
            dirpath="models/",
            filename="model-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # TODO: Decide if keeping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
        verbose=True,
    )
    callbacks.append(early_stopping)

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=("16-mixed" if get_device().type == "cuda" else "32"),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        # limit_train_batches=config.batch_limit if config.batch_limit else 1.0,
        # limit_val_batches=config.batch_limit if config.batch_limit else 1.0,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    setup_logging()
    freeze_support()
    main(config=config)
