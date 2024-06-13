import pytest
from sbx_word_prediction_kb_bert.annotations import load_predictor
from sbx_word_prediction_kb_bert.predictor import (
    TopKPredictor,
)


@pytest.fixture(scope="session")
def kb_bert_predictor() -> TopKPredictor:
    return load_predictor("3")
