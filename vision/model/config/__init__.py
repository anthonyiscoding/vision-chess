from argparse import Namespace
from types import SimpleNamespace
from dynaconf import Dynaconf
from vision.model.tokenizer import generate_all_possible_moves


def setup_hook(config):
    data = {"dynaconf_merge": False}
    if "num_kv_heads" not in config:
        data["num_kv_heads"] = config["num_heads"]

    if "hidden_dim" not in config:
        data["hidden_dim"] = config["emb_dim"] * 2

    if "head_dim" not in config:
        data["head_dim"] = config["emb_dim"] // config["num_heads"]

    if "batch_limit" not in config:
        data["batch_limit"] = None

    if "vocabulary_size" not in config:
        data["vocabulary_size"] = len(generate_all_possible_moves())

    return data


config = Dynaconf(
    envvar_prefix="VISION",
    settings_files=["default-settings.toml", ".secrets.toml", "custom-settings.toml"],
    environments=True,
    load_dotenv=True,
    env_switcher="VISION_ENV",
    post_hooks=setup_hook,
)

# TODO: Switch off of dynaconf this is ridiculous
config: dict = config.to_dict(env=config.env)
config.pop("LOAD_DOTENV", None)
config.pop("POST_HOOKS", None)
config = {key.lower(): value for key, value in config.items()}
config = Namespace(**config)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
