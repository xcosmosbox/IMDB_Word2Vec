def test_module_imports() -> None:
    """模块导入冒烟测试。"""
    import imdb_word2vec.config  # noqa: F401
    import imdb_word2vec.download  # noqa: F401
    import imdb_word2vec.preprocess  # noqa: F401
    import imdb_word2vec.feature_engineering  # noqa: F401
    import imdb_word2vec.fusion  # noqa: F401
    import imdb_word2vec.training  # noqa: F401
    import imdb_word2vec.cli  # noqa: F401


