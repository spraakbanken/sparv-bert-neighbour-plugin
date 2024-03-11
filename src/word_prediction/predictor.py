from transformers import (  # type: ignore [import-untyped]
    FillMaskPipeline,
)


class TopKPredictor:
    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

    def get_top_k_predictions(self, text: str, k=5) -> str:
        tokenized_inputs = self.tokenizer(text)
        if len(tokenized_inputs["input_ids"]) <= 512:
            return self._run_pipeline(text, k)
        if text.count("[MASK]") == 1:
            return self._run_pipeline_on_mask_context(text, k)
        raise RuntimeError(
            f"can't handle large input and multiple [MASK]: {len(tokenized_inputs['input_ids'])} tokens > 512 tokens"
        )

    def _run_pipeline_on_mask_context(self, text, k):
        start, end = self.compute_context(text)
        text_with_mask = text[start:end]
        return self._run_pipeline(text_with_mask, k)

    def compute_context(self, text):
        mask = text.find("[MASK]")
        lower = text[(mask - 210) : (mask - 190)].find(" ")
        higher = text[(mask + 190) : (mask + 210)].find(" ")
        start = mask - 210 + lower
        end = mask + 190 + higher
        return max(start, 0), min(end, len(text))

    def _run_pipeline(self, text, k) -> str:
        if predictions := self.pipeline(text, top_k=k):
            predictions_str = "|".join(
                f"{pred['token_str']}:{pred['score']}"  # type: ignore
                for pred in predictions
            )
            return f"|{predictions_str}|"
        else:
            return "|"
