from multiprocessing import freeze_support
import lightning as L
import git
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from vision.model import transformer as t
from vision.model.datamodule import ChessDataModule
from vision.model.config import config
from vision.utils import get_device


# TODO: Setup a --continue flag to allow model resuming
def main(config):
    data_module = ChessDataModule(config)
    model = t.ChessModel(config)

    # TODO: Make this not janky
    try:
        repo = git.Repo("../")
    except:
        repo = git.Repo("./")
    # Note: This is last commit, not all changes
    # TODO: Maybe force user to commit before running (allow override with flag)
    commit_id = repo.head.commit.hexsha[:7]

    date = datetime.now()
    logger = TensorBoardLogger(
        save_dir="logs", name=f"{config.env}-{date:%Y-%m-%d_%H-%M}-{commit_id}"
    )
    start_time = datetime.now()

    callbacks = []

    if config.save_model:
        # Lightning checkpoint callback - saves .ckpt files
        checkpoint_callback = ModelCheckpoint(
            dirpath="models/",
            filename=f"model-{start_time:%Y-%m-%d-%H:%M:%S}-{{epoch:02d}}-{{train_loss:.3f}}",
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
        log_every_n_steps=50,
        logger=logger,
        limit_train_batches=config.batch_limit if config.batch_limit else 1.0,
        limit_val_batches=(
            int(config.batch_limit * 0.1) if config.batch_limit else 1.0
        ),
        fast_dev_run=config.fast_dev_run if config.fast_dev_run else False,
        val_check_interval=0.5,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    freeze_support()
    main(config=config)
