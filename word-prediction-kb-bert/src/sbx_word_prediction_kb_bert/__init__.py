from dataclasses import dataclass
from typing import Optional, Tuple
from sparv.api import (  # type: ignore [import-untyped]
    annotator,
    Output,
    get_logger,
    Annotation,
    Config,
    SparvErrorMessage,
)

from transformers import (  # type: ignore [import-untyped]
    BertTokenizer,
    BertForMaskedLM,
)
from sbx_word_prediction_kb_bert.predictor import TopKPredictor

__description__ = "Calculating word predictions by mask a word in a BERT model."


__config__ = [
    Config(
        "sbx_word_prediction_kb_bert.num_predictions",
        description="The number of predictions to list",
        default=5,
    ),
    Config(
        "sbx_word_prediction_kb_bert.num_decimals",
        description="The number of decimals to round the score to",
        default=3,
    ),
]

__version__ = "0.5.2"

logger = get_logger(__name__)

TOK_SEP = " "


@dataclass
class HuggingfaceModel:
    model_name: str
    model_revision: str
    tokenizer_name: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    def tokenizer_name_and_revision(self) -> Tuple[str, str]:
        if tokenizer_name := self.tokenizer_name:
            return tokenizer_name, self.tokenizer_revision or "main"
        else:
            return self.model_name, self.model_revision


MODELS = {
    "kb-bert": HuggingfaceModel(
        model_name="KBLab/bert-base-swedish-cased",
        model_revision="c710fb8dff81abb11d704cd46a8a1e010b2b022c",
    )
}


@annotator("Word prediction tagging with a masked Bert model", language=["swe"])
def predict_words__kb_bert(
    out_prediction: Output = Output(
        "<token>:sbx_word_prediction_kb_bert.word-prediction--kb-bert",
        cls="word_prediction",
        description="Word predictions from masked BERT (format: '|<word>:<score>|...|)",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    num_predictions_str: str = Config("sbx_word_prediction_kb_bert.num_predictions"),
    num_decimals_str: str = Config("sbx_word_prediction_kb_bert.num_decimals"),
) -> None:
    logger.info("predict_words")
    try:
        num_predictions = int(num_predictions_str)
    except ValueError as exc:
        raise SparvErrorMessage(
            f"'sbx_word_prediction_kb_bert.num_predictions' must contain an 'int' got: '{num_predictions_str}'"
        ) from exc
    try:
        num_decimals = int(num_decimals_str)
    except ValueError as exc:
        raise SparvErrorMessage(
            f"'sbx_word_prediction_kb_bert.num_decimals' must contain an 'int' got: '{num_decimals_str}'"
        ) from exc
    tokenizer_name, tokenizer_revision = MODELS["kb-bert"].tokenizer_name_and_revision()

    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_name, revision=tokenizer_revision
    )
    model = BertForMaskedLM.from_pretrained(
        MODELS["kb-bert"].model_name, revision=MODELS["kb-bert"].model_revision
    )

    predictor = TopKPredictor(
        model=model,
        tokenizer=tokenizer,
        num_decimals=num_decimals,
    )

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
