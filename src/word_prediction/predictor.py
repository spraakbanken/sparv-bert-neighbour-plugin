from transformers import (  # type: ignore [import-untyped]
    FillMaskPipeline,
)


class HuggingFaceTopKPredictor:
    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

    def get_top_k_predictions(self, text: str, k=5) -> str:
        if predictions := self.pipeline(text, top_k=k):
            predictions_str = "|".join(
                f"{pred['token_str']}:{pred['score']}"  # type: ignore
                for pred in predictions
            )
            return f"|{predictions_str}|"
        else:
            return "|"
