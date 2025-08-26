import logging
from multiprocessing import freeze_support
import optuna
import optuna.trial as ot
from torch.utils.data import DataLoader
from vision.model.config import config
from vision.model.transformer import ChessModel
from vision.model.data import NpyDataset
from pgn_to_npy import list_npy_files
from vision.train import train
from vision.main import collate_fn, setup_logging
from vision.utils import get_device


logger = logging.getLogger(__name__)


# TODO: The way config currently works should be improved
def define_model_and_config(trial: ot.Trial):
    # config.transformer_layers = trial.suggest_int("transformer_layers", 2, 8)
    config.setenv("tune")
    config.batch_size = trial.suggest_int("batch_size", 2, 24, step=2)
    config.emb_dim = trial.suggest_categorical("emb_dim", [768, 1024, 2048, 4096])
    config.hidden_dim = config.emb_dim * 2
    config.head_dim = config.emb_dim // config.num_heads
    # config.qkv_bias = trial.suggest_categorical("qkv_bias", [True, False])
    config.learning_rate = trial.suggest_float("learning_rate", 3e-5, 1e-4, log=True)
    config.transformer_layers = trial.suggest_int("transformer_layers", 6, 12, step=2)

    return ChessModel(config), config


def objective(trial: ot.Trial):
    device = get_device()

    training_files = list_npy_files("data/training")
    validation_files = list_npy_files("data/validation")

    training_dataset = NpyDataset(training_files)
    validation_dataset = NpyDataset(validation_files)

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=2,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=2,
    )

    model, new_config = define_model_and_config(trial)
    accuracy = train(
        model=model,
        device=device,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        config=new_config,
        trial=trial,
    )

    return accuracy


if __name__ == "__main__":
    setup_logging()
    freeze_support()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, timeout=6000)

    failed_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.FAIL])
    pruned_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("Number of finished trials: %d", len(study.trials))
    logger.info("Number of failed trials: %d", len(failed_trials))
    logger.info("Number of pruned trials: %d", len(pruned_trials))
    logger.info("Number of complete trials: %d", len(complete_trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("Value: %s", str(trial.value))

    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info("    %s: %s", key, value)

    logger.info("All trials: %s", str(study.trials))
