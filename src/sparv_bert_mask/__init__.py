import token
from typing import Tuple
from sparv.api import annotator, Output, get_logger, Annotation

from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    TFBertForMaskedLM,
    FillMaskPipeline,
)
import torch
import numpy as np
import tensorflow as tf

__description__ = "Calculating word neighbours by mask a word in a BERT model."


logger = get_logger(__name__)

TOK_SEP = " "


@annotator(
    "Word neighbour tagging with a masked Bert model",
    language=["swe"],
)
def annotate_masked_bert(
    out_neighbour_torch: Output = Output(
        "<token>:sparv_bert_mask.transformer-neighbour-torch",
        cls="transformer_neighbour",
        description="Transformer neighbours from masked BERT (format: '|<word>:<score>|...|)",
    ),
    out_neighbour_tf: Output = Output(
        "<token>:sparv_bert_mask.transformer-neighbour-tf",
        cls="transformer_neighbour",
        description="Transformer neighbours from masked BERT (format: '|<word>:<score>|...|)",
    ),
    out_neighbour_hf: Output = Output(
        "<token>:sparv_bert_mask.transformer-neighbour-hf",
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

    torch_top_k_predictor = TorchTopKPredictor(tokenizer=tokenizer, model=model)
    tf_top_k_predictor = TFTopKPredictor(
        tokenizer=tokenizer, model=TFBertForMaskedLM.from_pretrained(model_name)
    )
    hf_top_k_predictor = HuggingFaceTopKPredictor(model=model, tokenizer=tokenizer)

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_neighbour_torch_annotation = word.create_empty_attribute()
    out_neighbour_tf_annotation = word.create_empty_attribute()
    out_neighbour_hf_annotation = word.create_empty_attribute()

    # sentences_to_tag = [
    #     [token_word[token_index] for token_index in sent] for sent in sentences
    # ]
    for sent in sentences:
        token_indices = list(sent)
        for token_index_to_mask in token_indices:
            sent_to_tag = TOK_SEP.join(
                "[MASK]"
                if token_index == token_index_to_mask
                else token_word[token_index]
                for token_index in sent
            )

            neighbours_scores = torch_top_k_predictor.get_top_k_predictions(sent_to_tag)

            out_neighbour_torch_annotation[token_index_to_mask] = neighbours_scores
            neighbours_scores = tf_top_k_predictor.get_top_k_predictions(sent_to_tag)
            # neighbours_scores = "|".join(
            #     f"{neighbour}:{score}"
            #     for neighbour, score in zip(neighbours.split(" "), scores)
            # )
            out_neighbour_tf_annotation[token_index_to_mask] = neighbours_scores
            neighbours_scores = hf_top_k_predictor.get_top_k_predictions(sent_to_tag)
            out_neighbour_hf_annotation[token_index_to_mask] = neighbours_scores

    logger.info("writing annotations")
    out_neighbour_torch.write(out_neighbour_torch_annotation)
    out_neighbour_tf.write(out_neighbour_tf_annotation)
    out_neighbour_hf.write(out_neighbour_hf_annotation)


class TorchTopKPredictor:
    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def get_top_k_predictions(self, text: str, k=5) -> str:
        tokenized_text = self.tokenizer.tokenize(text)
        masked_index = tokenized_text.index("[MASK]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        logger.info("torch text = '%s' probs = %s", text, probs)
        top_k_weights, top_k_indices = torch.topk(probs, k, sorted=True)

        # predicted_tokens = []
        # token_weights = []

        # for i, pred_idx in enumerate(top_k_indices):
        #     predicted_token = self.tokenizer.convert_ids_to_tokens([pred_idx])[0]
        #     predicted_tokens.append(predicted_token)
        #     token_weights.append(float(top_k_weights[i]))
        tokens_weights = (
            (
                self.tokenizer.convert_ids_to_tokens([pred_idx])[0],
                float(top_k_weights[i]),
            )
            for i, pred_idx in enumerate(top_k_indices)
        )
        result = "|".join(f"{word}:{score}" for word, score in tokens_weights)
        return f"|{result}|"


class TFTopKPredictor:
    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def get_top_k_predictions(self, text: str, k=5) -> str:
        tokenized_inputs = self.tokenizer(text, return_tensors="tf")
        outputs = self.model(**tokenized_inputs)

        mask_token = self.tokenizer.encode(self.tokenizer.mask_token)[1:-1]
        mask_index = np.where(tokenized_inputs.input_ids.numpy()[0] == mask_token)[0][0]

        # probs = tf.nn.softmax(outputs.logits[0, mask_index])
        # logger.info("tf text ='%s' probs = %s", text, probs)
        top_k_predictions = tf.math.top_k(outputs.logits, k, sorted=True)
        top_k_indices = top_k_predictions.indices[0].numpy()
        top_k_weights = top_k_predictions.values[0].numpy()
        decoded_output = self.tokenizer.batch_decode(top_k_indices)

        decoded_output_words = decoded_output[mask_index]

        result = "|".join(
            f"{word}:{score}"
            for word, score in zip(decoded_output_words.split(" "), top_k_weights)
        )
        return f"|{result}|"


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
