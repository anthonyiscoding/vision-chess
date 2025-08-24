import optuna
import optuna.trial as ot
import torch
from torch.utils.data import DataLoader
import vision.model.config.default as config
from vision.model.transformer import ChessModel
from vision.model.data import NpyDataset
from pgn_to_npy import list_npy_files
from vision.train import train
from vision.main import collate_fn


# TODO: The way config currently works should be improved
def define_model(trial: ot.Trial):
    config.transformer_layers = trial.suggest_int("transformer_layers", 2, 8)
    config.batch_size = trial.suggest_int("batch_size", 1, 7, step=2)
    config.emb_dim = trial.suggest_categorical(
        "emb_dim", [128, 256, 512, 768, 1024, 2048]
    )
    config.hidden_dim = config.emb_dim * 2
    config.head_dim = config.emb_dim // config.num_heads
    config.qkv_bias = trial.suggest_categorical("qkv_bias", [True, False])
    config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    return ChessModel(config)


def objective(trial: ot.Trial):
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

    model = define_model(trial)
    accuracy = train(
        model=model,
        device=device,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
    )

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[ot.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
