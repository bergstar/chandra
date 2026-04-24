from typing import List

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import scale_to_fit
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def apply_chat_template(processor, conversations):
    """Handle HF processor API differences across Transformers versions."""
    try:
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt",
                "padding": True,
            },
        )
    except TypeError as exc:
        if "processor_kwargs" not in str(exc):
            raise

        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )


def get_pad_token_id(model, eos_token_id):
    """Pick a stable pad token id so generation does not warn on open-ended runs."""
    pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.processor.tokenizer.pad_token_id
    if pad_token_id is None:
        tokenizer_eos = model.processor.tokenizer.eos_token_id
        if isinstance(tokenizer_eos, list):
            pad_token_id = tokenizer_eos[0]
        else:
            pad_token_id = tokenizer_eos
    if pad_token_id is None:
        if isinstance(eos_token_id, list):
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id
    return pad_token_id


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    bbox_scale: int = settings.BBOX_SCALE,
    **kwargs,
) -> List[GenerationResult]:
    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    conversations = [[process_batch_element(item)] for item in batch]

    inputs = apply_chat_template(model.processor, conversations)
    inputs = inputs.to(model.device)

    # Include both <|endoftext|> and <|im_end|> as stop tokens.
    # generation_config only has <|endoftext|>, but the model emits <|im_end|> at turn boundaries.
    eos_token_id = model.generation_config.eos_token_id
    im_end_id = model.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if im_end_id is not None and im_end_id not in eos_token_id:
        eos_token_id.append(im_end_id)
    pad_token_id = get_pad_token_id(model, eos_token_id)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_output_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = [
        GenerationResult(raw=out, token_count=len(ids), error=False)
        for out, ids in zip(output_text, generated_ids_trimmed)
    ]
    return results


def build_content(item: BatchInputItem):
    """Preserve the OCR path, but use text-first ordering for ad hoc prompts."""
    prompt = item.prompt
    prompt_type = item.prompt_type

    if not prompt:
        prompt = PROMPT_MAPPING[prompt_type]

    image = scale_to_fit(item.image)  # Guarantee max size
    if item.prompt is not None:
        return [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ]

    return [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]


def process_batch_element(item: BatchInputItem):
    return {"role": "user", "content": build_content(item)}


def load_model():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        raise ImportError(
            "HuggingFace backend requires additional dependencies. "
            "Install with: pip install chandra-ocr[hf]"
        )

    device_map = "auto"
    if settings.TORCH_DEVICE:
        device_map = {"": settings.TORCH_DEVICE}

    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
    }
    if settings.TORCH_ATTN:
        kwargs["attn_implementation"] = settings.TORCH_ATTN

    model = AutoModelForImageTextToText.from_pretrained(
        settings.MODEL_CHECKPOINT, **kwargs
    )
    model = model.eval()
    processor = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    processor.tokenizer.padding_side = "left"
    model.processor = processor
    return model
