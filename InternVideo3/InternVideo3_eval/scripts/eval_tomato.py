import argparse
import json
import os
import re
import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# ==================== 数据路径配置 ====================
DATA_ROOT = os.environ.get("DATA_DIR", "/path/to/data/TOMATO")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
DATA_DIR = os.path.join(DATA_ROOT, "data")
SPLITS = ["count", "direction", "rotation", "shape_trend", "velocity_frequency", "visual_cues"]
# =====================================================

MCQ_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "{question}\n"
    "{options}\n"
    "Answer with the option letter only."
)


def get_rank_and_world_size():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def parse_answer(text):
    """Extract the option letter from model output."""
    text = text.strip()
    m = re.match(r"^([A-Z])", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"(?:answer is|answer:)?\s*([A-Z])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return text.strip()[0].upper() if text.strip() else ""


def build_samples(data_dir):
    """Load all parquet splits and return list of sample dicts."""
    samples = []
    for split in SPLITS:
        parquet_files = [f for f in os.listdir(data_dir) if f.startswith(split + "-")]
        for pf in parquet_files:
            df = pd.read_parquet(os.path.join(data_dir, pf))
            for idx, row in df.iterrows():
                options = list(row["options"])
                answer_idx = int(row["answer"])
                # Build option letters: A, B, C, ...
                option_letters = [chr(ord('A') + i) for i in range(len(options))]
                answer_letter = option_letters[answer_idx]
                options_text = "\n".join(
                    f"{letter}. {opt}" for letter, opt in zip(option_letters, options)
                )

                # Video path: videos/{demonstration_type}/{key}.mp4
                demo_type = row["demonstration_type"]
                key = row["key"]
                video_path = os.path.join(VIDEO_DIR, demo_type, f"{key}.mp4")

                samples.append({
                    "video_path": video_path,
                    "question_id": f"{split}_{idx}",
                    "question": row["question"],
                    "options_text": options_text,
                    "answer": answer_letter,
                    "split": split,
                    "motion_type": row.get("motion_type", split),
                })
    return samples


def run_inference(model, processor, video_path, prompt, fps, min_pixels, max_pixels):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    processor.video_processor.size = {"longest_edge": max_pixels * 512, "shortest_edge": min_pixels * 32}

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        fps=4,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    output = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, output)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return response


def evaluate(model, processor, args, rank, world_size):
    samples = build_samples(DATA_DIR)
    my_samples = samples[rank::world_size]

    output_path = os.path.join(args.output_dir, f"tomato_rank{rank}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    # resume
    done_keys = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                done_keys.add(item["question_id"])

    fout = open(output_path, "a")

    for sample in tqdm(
        my_samples, desc=f"[TOMATO][rank{rank}]", disable=(rank != 0)
    ):
        if sample["question_id"] in done_keys:
            continue

        if not os.path.exists(sample["video_path"]):
            print(f"WARNING [rank{rank}]: video not found: {sample['video_path']}")
            continue

        prompt = MCQ_PROMPT.format(
            question=sample["question"],
            options=sample["options_text"],
        )

        try:
            response = run_inference(
                model, processor, sample["video_path"], prompt,
                args.fps, args.min_pixels, args.max_pixels,
            )
            pred = parse_answer(response)
        except Exception as e:
            print(f"ERROR [rank{rank}] qid={sample['question_id']}: {e}")
            response = ""
            pred = ""

        result = {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "pred": pred,
            "correct": pred == sample["answer"],
            "split": sample["split"],
            "response": response,
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()

    if dist.is_initialized():
        dist.barrier()

    # rank 0 merges and reports
    if rank == 0:
        merged_path = os.path.join(args.output_dir, "tomato_results.jsonl")
        total, correct = 0, 0
        split_stats = {}  # split -> [total, correct]

        with open(merged_path, "w") as fmerge:
            for r in range(world_size):
                rpath = os.path.join(args.output_dir, f"tomato_rank{r}.jsonl")
                if not os.path.exists(rpath):
                    continue
                with open(rpath) as f:
                    for line in f:
                        fmerge.write(line)
                        item = json.loads(line)
                        total += 1
                        c = 1 if item["correct"] else 0
                        correct += c

                        s = item.get("split", "unknown")
                        if s not in split_stats:
                            split_stats[s] = [0, 0]
                        split_stats[s][0] += 1
                        split_stats[s][1] += c

        print(f"\n{'='*50}")
        print(f"TOMATO | Total: {total}")
        print(f"  Overall Acc: {correct / max(total, 1):.4f} ({correct}/{total})")
        for s in sorted(split_stats.keys()):
            tot, cor = split_stats[s]
            print(f"  {s:>25s}: {cor / max(tot, 1):.4f} ({cor}/{tot})")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str,
                        default=None, help="Path to processor (defaults to model_path if not set)")
    parser.add_argument("--output_dir", type=str,
                        default="./logs/tomato_results")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--min_pixels", type=int, default=256 * 4 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=512 * 2 * 32 * 32)
    args = parser.parse_args()

    rank, world_size = 0, 1
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

    device = f"cuda:{rank}"

    if rank == 0:
        print(f"Loading model from {args.model_path} ...")
        print(f"World size: {world_size}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.processor_path, trust_remote_code=True,
    )

    evaluate(model, processor, args, rank, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
