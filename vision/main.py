from multiprocessing import freeze_support
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from vision.model import transformer as t
from vision.model.datamodule import ChessDataModule
from vision.model.config import config
from vision.utils import get_device


def main(config):
    data_module = ChessDataModule(config)
    model = t.ChessModel(config)
    logger = TensorBoardLogger(save_dir="logs", name=config.env)

    callbacks = []

    if config.save_model:
        # Lightning checkpoint callback - saves .ckpt files
        checkpoint_callback = ModelCheckpoint(
            dirpath="models/",
            filename="model-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last="link",
        )
        callbacks.append(checkpoint_callback)

    early_stopping_val_loss = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
        stopping_threshold=1e-2,
        verbose=True,
    )
    callbacks.append(early_stopping_val_loss)

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=("16-mixed" if get_device().type == "cuda" else "32"),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=logger,
        limit_train_batches=config.batch_limit if config.batch_limit else 1.0,
        limit_val_batches=(
            max(int(config.batch_limit * 0.1), 2) if config.batch_limit else 1.0
        ),
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    freeze_support()
    main(config=config)
