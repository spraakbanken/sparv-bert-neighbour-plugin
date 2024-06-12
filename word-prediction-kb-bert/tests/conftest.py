import pytest
from sbx_word_prediction_kb_bert import (
    TopKPredictor,
)


@pytest.fixture(scope="session")
def kb_bert_predictor() -> TopKPredictor:
    return TopKPredictor()
