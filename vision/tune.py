import copy
import logging
import sys
import optuna
import optuna.trial as ot
import lightning as L
from multiprocessing import freeze_support
from vision.model.config import config
from vision.model.transformer import ChessModel
from vision.model.datamodule import ChessDataModule
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback

from vision.utils import get_device


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


stdout_logger = logging.getLogger(__name__)


def objective(trial: ot.Trial):
    L.seed_everything(123)
    new_config = copy.deepcopy(config)

    new_config.batch_size = trial.suggest_int("batch_size", 2, 24, step=2)
    new_config.emb_dim = trial.suggest_categorical("emb_dim", [768, 1024, 2048, 4096])
    new_config.hidden_dim = new_config.emb_dim * 2
    new_config.head_dim = new_config.emb_dim // new_config.num_heads
    new_config.learning_rate = trial.suggest_float(
        "learning_rate", 3e-5, 1e-3, log=True
    )
    new_config.transformer_layers = trial.suggest_int(
        "transformer_layers", 2, 12, step=2
    )
    # config.qkv_bias = trial.suggest_categorical("qkv_bias", [True, False])

    model = ChessModel(new_config)
    data_module = ChessDataModule(new_config)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=False, mode="min"
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    logger = TensorBoardLogger(save_dir="logs", name="optuna_trial_{trial.number}")

    callbacks = [pruning_callback, early_stop_callback]
    trainer = L.Trainer(
        max_epochs=new_config.num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=("16-mixed" if get_device().type == "cuda" else "32"),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=logger,
        limit_train_batches=new_config.batch_limit if new_config.batch_limit else 1.0,
        limit_val_batches=(
            max(int(new_config.batch_limit * 0.1), 2) if new_config.batch_limit else 1.0
        ),
        fast_dev_run=False,
    )
    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    setup_logging()
    freeze_support()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=6000)

    failed_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.FAIL])
    pruned_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.COMPLETE])

    stdout_logger.info("Study statistics: ")
    stdout_logger.info("Number of finished trials: %d", len(study.trials))
    stdout_logger.info("Number of failed trials: %d", len(failed_trials))
    stdout_logger.info("Number of pruned trials: %d", len(pruned_trials))
    stdout_logger.info("Number of complete trials: %d", len(complete_trials))

    stdout_logger.info("Best trial:")
    trial = study.best_trial

    stdout_logger.info("Value: %s", str(trial.value))

    stdout_logger.info("Params: ")
    for key, value in trial.params.items():
        stdout_logger.info("    %s: %s", key, value)

    stdout_logger.info("All trials: %s", str(study.trials))
