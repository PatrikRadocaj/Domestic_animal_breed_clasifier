from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_config_resolves():
    conf_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert cfg.project_name == "animal-breed-detection"
    assert cfg.paths.artifacts_dir == "artifacts"
