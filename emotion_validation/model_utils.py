from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_story_activation(
    model,
    tokenizer,
    text: str,
    layer: int,
    token_offset: int = 50,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """
    Returns the mean float32 residual-stream vector across all token positions
    from token_offset onward, captured at the output of model.model.layers[layer].

    Returns None if the text tokenizes to fewer tokens than token_offset.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if inputs.input_ids.shape[1] <= token_offset:
        return None

    captured: list[torch.Tensor] = []

    def _hook(_module, _input, output):
        # output[0]: (batch=1, seq_len, d_model) bfloat16
        hidden = output[0][0, token_offset:, :].detach().float()
        captured.append(hidden.mean(dim=0).cpu())

    handle = model.model.layers[layer].register_forward_hook(_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    return captured[0] if captured else None
