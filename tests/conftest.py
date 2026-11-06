import pytest
import spacy


def model_available(name: str) -> bool:
    return spacy.util.is_package(name)


requires_sm = pytest.mark.skipif(not model_available("en_core_web_sm"), reason="en_core_web_sm not installed")
