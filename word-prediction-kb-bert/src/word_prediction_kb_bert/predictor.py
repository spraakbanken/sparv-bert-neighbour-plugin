from transformers import (  # type: ignore [import-untyped]
    FillMaskPipeline,
)

SCORE_FORMATS = {
    1: ("{:.1f}", lambda s: s.endswith(".0")),
    2: ("{:.2f}", lambda s: s.endswith(".00")),
    3: ("{:.3f}", lambda s: s.endswith(".000")),
    4: ("{:.4f}", lambda s: s.endswith(".0000")),
    5: ("{:.5f}", lambda s: s.endswith(".00000")),
    6: ("{:.6f}", lambda s: s.endswith(".000000")),
    7: ("{:.7f}", lambda s: s.endswith(".0000000")),
    8: ("{:.8f}", lambda s: s.endswith(".00000000")),
    9: ("{:.9f}", lambda s: s.endswith(".000000000")),
    10: ("{:.10f}", lambda s: s.endswith(".0000000000")),
}


class TopKPredictor:
    def __init__(self, *, tokenizer, model, num_decimals: int = 3) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.num_decimals = num_decimals
        self.pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

    def get_top_k_predictions(self, text: str, k: int = 5) -> str:
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
            collect_token_and_score = (
                (pred["token_str"], pred["score"])  # type: ignore
                for pred in predictions
            )
            score_format, score_pred = SCORE_FORMATS[self.num_decimals]
            format_scores = (
                (token, score_format.format(score))
                for token, score in collect_token_and_score
            )
            filter_out_zero_scores = (
                (token, score)
                for token, score in format_scores
                if not score_pred(score)
            )
            predictions_str = "|".join(
                f"{token}:{score}" for token, score in filter_out_zero_scores
            )

            return f"|{predictions_str}|" if predictions_str else "|"
        else:
            return "|"
