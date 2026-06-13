from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Unquantized bf16. gemma-3-4b (~8-9 GB) fits on a single 24 GB A10G with room.
    # Quantization is what pushed the BOS-token activation over into inf, which the
    # attention sink then spread to every position as NaN; full bf16 keeps it finite.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _get_layer(model, layer: int):
    # Locate the decoder layer list. The path varies by transformers version and
    # whether Gemma-3 loaded as the multimodal Gemma3ForConditionalGeneration
    # (layers under model.model.language_model) or a plain text causal LM.
    candidates = (
        lambda m: m.language_model.layers,         # text model exposed directly
        lambda m: m.model.language_model.layers,   # Gemma3ForConditionalGeneration -> Gemma3Model -> text
        lambda m: m.model.layers,                  # plain causal LM
        lambda m: m.layers,                        # bare decoder
    )
    for get in candidates:
        try:
            return get(model)[layer]
        except (AttributeError, TypeError, IndexError):
            continue
    raise AttributeError("Could not locate decoder layers on this model structure")


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
    from token_offset onward, captured at the output of the transformer layer.

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

    handle = _get_layer(model, layer).register_forward_hook(_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    return captured[0] if captured else None
