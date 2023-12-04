from sparv.api import annotator, Output, get_logger, Annotation

from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    FillMaskPipeline,
)

__description__ = "Calculating word neighbours by mask a word in a BERT model."


logger = get_logger(__name__)

TOK_SEP = " "


@annotator(
    "Word neighbour tagging with a masked Bert model",
    language=["swe"],
)
def annotate_masked_bert(
    out_neighbour: Output = Output(
        "<token>:sparv_bert_mask.transformer-neighbour",
        cls="transformer_neighbour",
        description="Transformer neighbours from masked BERT (format: '|<word>:<score>|...|)",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
) -> None:
    logger.info("annotate_masked_bert")

    tokenizer_name = "KB/bert-base-swedish-cased"
    model_name = "KB/bert-base-swedish-cased"

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    hf_top_k_predictor = HuggingFaceTopKPredictor(model=model, tokenizer=tokenizer)

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_neighbour_annotation = word.create_empty_attribute()

    for sent in sentences:
        token_indices = list(sent)
        for token_index_to_mask in token_indices:
            sent_to_tag = TOK_SEP.join(
                "[MASK]"
                if token_index == token_index_to_mask
                else token_word[token_index]
                for token_index in sent
            )

            neighbours_scores = hf_top_k_predictor.get_top_k_predictions(sent_to_tag)
            out_neighbour_annotation[token_index_to_mask] = neighbours_scores

    logger.info("writing annotations")
    out_neighbour.write(out_neighbour_annotation)


class HuggingFaceTopKPredictor:
    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

    def get_top_k_predictions(self, text: str, k=5) -> str:
        if predictions := self.pipeline(text, top_k=k):
            predictions_str = "|".join(
                f"{pred['token_str']}:{pred['score']}" for pred in predictions
            )
            return f"|{predictions_str}|"
        else:
            return "|"
