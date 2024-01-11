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
    FillMaskPipeline,
)

__description__ = "Calculating word neighbours by mask a word in a BERT model."


__config__ = [
    Config(
        "sparv_bert_neighbour.model",
        description="Huggingface pretrained model name",
        default="KB/bert-base-swedish-cased",
    ),
    Config(
        "sparv_bert_neighbour.tokenizer",
        description="HuggingFace pretrained tokenizer name",
        default="KB/bert-base-swedish-cased",
    ),
    Config(
        "sparv_bert_neighbour.num_neighbours",
        description="The number of neighbours to list",
        default=5,
    ),
]

__version__ = "0.1.2"

logger = get_logger(__name__)

TOK_SEP = " "


@annotator(
    "Word neighbour tagging with a masked Bert model",
)
def annotate_masked_bert(
    out_neighbour: Output = Output(
        "<token>:sparv_bert_neighbour.transformer-neighbour",
        cls="transformer_neighbour",
        description="Transformer neighbours from masked BERT (format: '|<word>:<score>|...|)",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    model_name: str = Config("sparv_bert_neighbour.model"),
    tokenizer_name: str = Config("sparv_bert_neighbour.tokenizer"),
    num_neighbours_str: str = Config("sparv_bert_neighbour.num_neighbours"),
) -> None:
    logger.info("annotate_masked_bert")
    try:
        num_neighbours = int(num_neighbours_str)
    except ValueError as exc:
        raise SparvErrorMessage(
            f"'sparv_bert_neighbour.num_neighbours' must contain an 'int' got: '{num_neighbours_str}'"
        ) from exc
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    hf_top_k_predictor = HuggingFaceTopKPredictor(model=model, tokenizer=tokenizer)

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_neighbour_annotation = word.create_empty_attribute()

    logger.progress(total=len(sentences))
    for sent in sentences:
        logger.progress()
        token_indices = list(sent)
        for token_index_to_mask in token_indices:
            sent_to_tag = TOK_SEP.join(
                "[MASK]"
                if token_index == token_index_to_mask
                else token_word[token_index]
                for token_index in sent
            )

            neighbours_scores = hf_top_k_predictor.get_top_k_predictions(
                sent_to_tag, k=num_neighbours
            )
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
