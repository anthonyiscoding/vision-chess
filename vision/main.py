from multiprocessing import freeze_support
from lightning.pytorch.cli import LightningCLI
import git
from datetime import datetime
from vision.model.transformer import ChessModel
from vision.model.datamodule import ChessDataModule


# TODO: Stop overriding user specified values
class CustomLightningCLI(LightningCLI):
    """Custom LightningCLI that adds commit_id and date to logger name and checkpoint filename (overrides user specified values)."""

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
    cli = CustomLightningCLI(ChessModel, ChessDataModule)


if __name__ == "__main__":
    freeze_support()
    main()
