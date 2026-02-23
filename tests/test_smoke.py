import minimal10digittransformer


def test_version() -> None:
    assert minimal10digittransformer.__version__ == "0.1.0"


def test_consts_imported() -> None:
    assert minimal10digittransformer.PROJECT_ROOT.exists()
    assert minimal10digittransformer.DATA_DIR.is_absolute()
