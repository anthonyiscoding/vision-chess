import torch
from lightning.pytorch.cli import LightningCLI
import git
from datetime import datetime
from pathlib import Path
from vision.model.transformer import ChessModel
from vision.model.datamodule import ChessDataModule


# TODO: Stop overriding user specified values
class CustomLightningCLI(LightningCLI):
    """Custom LightningCLI that adds commit_id and date to logger name and checkpoint filename (overrides user specified values)."""

    def _parse_ckpt_path(self) -> None:
        """Override to selectively preserve config parameters while loading from checkpoint."""
        if not self.config.get("subcommand"):
            return

        ckpt_path = self.config[self.config.subcommand].get("ckpt_path")
        if not (ckpt_path and Path(ckpt_path).is_file()):
            return

        preserved_params = {}
        if hasattr(self.config[self.config.subcommand], "model"):
            model_config = self.config[self.config.subcommand].model
            # List of parameters to preserve from config (add/remove as needed)
            preserve_keys = [
                "learning_rate",
                "scheduler_patience",
                "reduce_lr_by",
            ]

            for key in preserve_keys:
                if hasattr(model_config, key):
                    preserved_params[key] = getattr(model_config, key)

        super()._parse_ckpt_path()

        # Restore preserved parameters
        if preserved_params and hasattr(self.config[self.config.subcommand], "model"):
            model_config = self.config[self.config.subcommand].model
            for key, value in preserved_params.items():
                setattr(model_config, key, value)

    def add_arguments_to_parser(self, parser):
        # Add custom arguments if needed
        super().add_arguments_to_parser(parser)

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()

        # TODO: Make this not janky
        try:
            repo = git.Repo("../")
        except:
            repo = git.Repo("./")

        commit_id = repo.head.commit.hexsha[:7]
        date = datetime.now()

        env_name = getattr(self.config, "env", "default")

        # TODO: Figure out the correct routes to these variables
        # Updates logger name with commit and date info
        try:
            logger_name = f"{env_name}-{date:%Y-%m-%d_%H-%M}-{commit_id}"
            self.config.trainer.logger.init_args["name"] = logger_name
        except:
            pass

        # Updates ModelCheckpoint filename if present
        try:
            filename = f"model-{date:%Y-%m-%d-%H-%M-%S}-{{epoch:02d}}-{{train_loss:.3f}}-commit={commit_id}"
            self.config.trainer.callbacks.init_args.filename = filename
        except:
            pass


def main():
    torch.set_float32_matmul_precision("medium")
    cli = CustomLightningCLI(
        ChessModel,
        ChessDataModule,
    )


if __name__ == "__main__":
    main()
