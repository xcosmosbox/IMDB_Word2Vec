from pathlib import Path

from imdb_word2vec.config import PathConfig


def test_pathconfig_ensure(tmp_path: Path) -> None:
    """确保路径创建逻辑可用，不污染真实目录。"""
    paths = PathConfig(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        logs_dir=tmp_path / "logs",
        artifacts_dir=tmp_path / "artifacts",
    )
    paths.ensure()

    assert paths.data_dir.exists()
    assert paths.cache_dir.exists()
    assert paths.logs_dir.exists()
    assert paths.artifacts_dir.exists()


