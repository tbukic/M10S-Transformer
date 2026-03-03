import minimal10digittransformer


def test_version() -> None:
    assert minimal10digittransformer.__version__ == "0.1.0"


def test_model_importable() -> None:
    from minimal10digittransformer.model.qwen3 import Qwen3AdditionModel
    assert Qwen3AdditionModel is not None


def test_data_importable() -> None:
    from minimal10digittransformer.data.addition import encode, generate_batch
    assert encode is not None
    assert generate_batch is not None
