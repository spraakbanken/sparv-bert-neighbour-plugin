from typing import Optional

from sparv import api as sparv_api  # type: ignore [import-untyped]
from sparv.api import (  # type: ignore [import-untyped]
    Annotation,
    Config,
    Output,
)

from sbx_word_prediction_kb_bert.constants import PROJECT_NAME
from sbx_word_prediction_kb_bert.predictor import TopKPredictor

logger = sparv_api.get_logger(__name__)
TOK_SEP = " "


def load_predictor(num_decimals_str: str) -> TopKPredictor:
    try:
        num_decimals = int(num_decimals_str)
    except ValueError as exc:
        raise sparv_api.SparvErrorMessage(
            f"'{PROJECT_NAME}.num_decimals' must contain an 'int' got: '{num_decimals_str}'"  # noqa: E501
        ) from exc

    return TopKPredictor(num_decimals=num_decimals)


@sparv_api.annotator(
    "Word prediction tagging with a masked Bert model",
    language=["swe"],
    preloader=load_predictor,
    preloader_params=["num_decimals_str"],
    preloader_target="predictor_preloaded",
)
def predict_words__kb_bert(
    out_prediction: Output = Output(
        f"<token>:{PROJECT_NAME}.word-prediction--kb-bert",
        description="Word predictions from masked BERT (format: '|<word>:<score>|...|)",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    num_predictions_str: str = Config(f"{PROJECT_NAME}.num_predictions"),
    num_decimals_str: str = Config(f"{PROJECT_NAME}.num_decimals"),
    predictor_preloaded: Optional[TopKPredictor] = None,
) -> None:
    logger.info("predict_words")
    try:
        num_predictions = int(num_predictions_str)
    except ValueError as exc:
        raise sparv_api.SparvErrorMessage(
            f"'{PROJECT_NAME}.num_predictions' must contain an 'int' got: '{num_predictions_str}'"  # noqa: E501
        ) from exc

    predictor = predictor_preloaded or load_predictor(num_decimals_str)

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_prediction_annotation = word.create_empty_attribute()

    run_word_prediction(
        predictor=predictor,
        num_predictions=num_predictions,
        sentences=sentences,
        token_word=token_word,
        out_prediction_annotations=out_prediction_annotation,
    )

    logger.info("writing annotations")
    out_prediction.write(out_prediction_annotation)


def run_word_prediction(
    predictor: TopKPredictor,
    num_predictions: int,
    sentences,
    token_word: list,
    out_prediction_annotations,
) -> None:
    logger.info("run_word_prediction")

    logger.progress(total=len(sentences))  # type: ignore
    for sent in sentences:
        logger.progress()  # type: ignore
        token_indices = list(sent)
        for token_index_to_mask in token_indices:
            sent_to_tag = TOK_SEP.join(
                (
                    "[MASK]"
                    if token_index == token_index_to_mask
                    else token_word[token_index]
                )
                for token_index in sent
            )

            predictions_scores = predictor.get_top_k_predictions(
                sent_to_tag, k=num_predictions
            )
            out_prediction_annotations[token_index_to_mask] = predictions_scores
