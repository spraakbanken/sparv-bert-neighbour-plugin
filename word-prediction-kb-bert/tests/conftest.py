import pytest
from sbx_word_prediction_kb_bert import (
    MODELS,
    TopKPredictor,
)
from transformers import (  # type: ignore [import-untyped]
    BertTokenizer,
    BertForMaskedLM,
)


@pytest.fixture(scope="session")
def kb_bert_predictor() -> TopKPredictor:
    tokenizer_name, tokenizer_revision = MODELS["kb-bert"].tokenizer_name_and_revision()
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_name, revision=tokenizer_revision
    )
    model = BertForMaskedLM.from_pretrained(
        MODELS["kb-bert"].model_name, revision=MODELS["kb-bert"].model_revision
    )
    return TopKPredictor(model=model, tokenizer=tokenizer)
