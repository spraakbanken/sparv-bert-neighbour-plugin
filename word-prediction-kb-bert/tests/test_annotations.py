import pytest
from sbx_word_prediction_kb_bert.annotations import (
    load_predictor,
    predict_words__kb_bert,
)
from sparv.api import SparvErrorMessage  # type: ignore [import-untyped]
from sparv_pipeline_testing import MemoryOutput, MockAnnotation


def test_load_predictor_not_int_raises() -> None:
    with pytest.raises(SparvErrorMessage):
        load_predictor("not an int")


def test_predict_words__kb_bert__bad_int_raises() -> None:
    with pytest.raises(SparvErrorMessage):
        predict_words__kb_bert(
            out_prediction=MemoryOutput(),
            word=MockAnnotation(),
            sentence=MockAnnotation(),
            num_predictions_str="not-an-int",
            num_decimals_str="2",
        )


def test_predict_words__kb_bert(snapshot) -> None:
    output: MemoryOutput = MemoryOutput()

    word = MockAnnotation(
        name="<token:word>", values=["Han", "åt", "glassen", "utanför", "kiosken", "."]
    )
    sentence = MockAnnotation(
        name="<sentence>", children={"<token:word>": [[0, 1, 2, 3, 4, 5]]}
    )

    predict_words__kb_bert(
        output, word, sentence, num_predictions_str="5", num_decimals_str="2"
    )

    assert output.values == snapshot
