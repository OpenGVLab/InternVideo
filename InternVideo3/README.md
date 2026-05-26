# InternVideo3: Multimodal Contextual Reasoning via Efficient Long-Horizon Agents

## Introduction

InternVideo3 is a multimodal large language model designed for long-horizon video understanding and agentic reasoning. It introduces **Multimodal Contextual Reasoning (MCR)**, an efficient formulation that unifies perception, planning, tool use, self-reflection, and memory within a single shared context, enabling recursive multi-step reasoning over long videos.

### Key Features

- **M²LA (Multimodal Multi-head Latent Attention):** A KV-cache-efficient attention architecture that reduces memory footprint via low-rank latent factorization, enabling long-context reasoning (up to 256K tokens) without dropping tokens.
- **Long-Video Understanding:** Trained with a short-to-long curriculum (up to 2048 frames at 4fps), supporting hour-long video comprehension.
- **Agentic Video Reasoning:** Built-in support for recursive perception-action loops with tool use (temporal grounding, ASR, web search, video segmentation) and self-verification.
- **Advanced Post-Training:** Combines rule-based group sequence policy optimization (R-GSPO) and on-policy distillation from Qwen3-235B for improved temporal reasoning.

### Architecture

| Component | Details |
|-----------|---------|
| Vision Encoder | 27-layer ViT, hidden_size=1152, patch_size=16, temporal_patch_size=2 |
| Language Model | 36-layer, hidden_size=4096, 32 attention heads |
| KV Latent Rank | 896 per layer |
| Max Context | 262,144 tokens |
| Precision | BFloat16 |

## Model Zoo

| Model | HuggingFace |
|-------|-------------|
| InternVideo3-8B-Instruct | [yanziang/InternVideo3-8B-Instruct](https://huggingface.co/yanziang/InternVideo3-8B-Instruct) |

## Quickstart

### Requirements

```bash
pip install transformers>=4.57.3 torch qwen-vl-utils
```

### Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "OpenGVLab/InternVideo3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)
```

### Text-only Conversation

```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Please introduce yourself."}],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = processor(text=text, images=None, videos=None, do_resize=False, return_tensors="pt")
inputs = inputs.to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### Video Understanding

```python
video_path = "your_video.mp4"

fps = 1
min_pixels = 128 * 32 * 32
max_pixels = 128 * 32 * 32

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": fps},
            {"type": "text", "text": "Please describe this video in detail."},
        ],
    }
]

processor.video_processor.size = {
    "longest_edge": max_pixels * max_frames,
    "shortest_edge": min_pixels * min_frames,
}

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    fps=fps,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### Image Understanding

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "your_image.jpg"},
            {"type": "text", "text": "Please describe this image in detail."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = processor(text=text, images=images, videos=None, do_resize=False, return_tensors="pt")
inputs = inputs.to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Training Pipeline

1. **Continued Pretraining (CPT):** Recovers language ability and aligns vision features after M²LA conversion, using a mixture of text, image-text pairs, and video captions.
2. **Short-to-Long SFT:** Two-stage curriculum — Stage 1 at 2fps/512 frames (32K tokens), Stage 2 at 4fps/2048 frames (256K tokens).
3. **R-GSPO:** Rule-based reinforcement learning on temporal grounding (IoU reward) and video QA (correctness reward) to improve temporal reasoning.
4. **On-Policy Distillation:** Transfers capabilities from Qwen3-235B on samples where the student underperforms, using reverse-KL on student-sampled trajectories.

## Evaluation

See the [InternVideo3_eval](InternVideo3_eval) directory for evaluation scripts and benchmarks.

## Citation

```bibtex
@article{internvideo3,
  title={InternVideo3: Multimodal Contextual Reasoning via Efficient Long-Horizon Agents},
  author={InternVideo Team},
  year={2025}
}
```

## License

This project is released under the Apache 2.0 License.
