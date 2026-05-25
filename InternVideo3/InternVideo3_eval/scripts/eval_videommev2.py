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
DATA_ROOT = os.environ.get("DATA_DIR", "/path/to/data/VideoMME-v2")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
SUBTITLE_DIR = os.path.join(DATA_ROOT, "subtitle")
PARQUET_PATH = os.path.join(DATA_ROOT, "test-00000-of-00001.parquet")
# =====================================================

MCQ_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Question: {question}\nOptions:\n{options}\n"
    "Answer with the option letter only."
)

MCQ_PROMPT_WITH_SUB = (
    "Select the best answer to the following multiple-choice question based on the video. "
    "The subtitles of the video are also provided below.\n"
    "Subtitles:\n{subtitles}\n\n"
    "Question: {question}\nOptions:\n{options}\n"
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
    # Try to match a single letter at start
    m = re.match(r"^([A-H])", text.upper())
    if m:
        return m.group(1)
    # Fallback: find any letter preceded by common patterns
    m = re.search(r"(?:answer is|answer:)?\s*([A-H])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return text.strip()[0].upper() if text.strip() else ""


def load_subtitles(video_id):
    """Load subtitles from jsonl file."""
    sub_path = os.path.join(SUBTITLE_DIR, f"{video_id}.jsonl")
    if not os.path.exists(sub_path):
        return ""
    lines = []
    with open(sub_path) as f:
        for line in f:
            item = json.loads(line)
            lines.append(item["text"])
    return " ".join(lines)


def build_samples(parquet_path):
    """Load parquet and return list of sample dicts."""
    df = pd.read_parquet(parquet_path)
    samples = []
    for _, row in df.iterrows():
        options_list = row["options"]
        options_str = "\n".join(options_list)
        samples.append({
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "options_str": options_str,
            "answer": row["answer"],
            "group_type": row["group_type"],
            "level": row.get("level", None),
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
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
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
    samples = build_samples(PARQUET_PATH)
    my_samples = samples[rank::world_size]

    for use_subtitle in ([False, True] if args.with_subtitle else [False]):
        suffix = "w_sub" if use_subtitle else "wo_sub"
        # suffix = "w_sub"
        print(f"Evaluating {suffix} ...")
        output_path = os.path.join(args.output_dir, f"videommev2_{suffix}_rank{rank}.jsonl")
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
            my_samples, desc=f"[VideoMME-v2 {suffix}][rank{rank}]", disable=(rank != 0)
        ):
            if sample["question_id"] in done_keys:
                continue

            video_path = os.path.join(VIDEO_DIR, f"{sample['video_id']}.mp4")
            if not os.path.exists(video_path):
                print(f"WARNING [rank{rank}]: video not found: {video_path}")
                continue

            if use_subtitle:
                subtitles = load_subtitles(sample["video_id"])
                prompt = MCQ_PROMPT_WITH_SUB.format(
                    subtitles=subtitles,
                    question=sample["question"],
                    options=sample["options_str"],
                )
            else:
                prompt = MCQ_PROMPT.format(
                    question=sample["question"],
                    options=sample["options_str"],
                )

            try:
                response = run_inference(
                    model, processor, video_path, prompt,
                    args.fps, args.min_pixels, args.max_pixels,
                )
                pred = parse_answer(response)
            except Exception as e:
                print(f"ERROR [rank{rank}] {sample['question_id']}: {e}")
                response = ""
                pred = ""

            result = {
                "video_id": sample["video_id"],
                "question_id": sample["question_id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "pred": pred,
                "correct": pred == sample["answer"],
                "group_type": sample["group_type"],
                "level": sample["level"],
                "response": response,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

        fout.close()

        if dist.is_initialized():
            dist.barrier()

        # rank 0 merges and reports
        if rank == 0:
            merged_path = os.path.join(args.output_dir, f"videommev2_{suffix}_results.jsonl")
            total, correct = 0, 0
            type_stats = {}  # group_type -> [total, correct]
            level_stats = {}  # level -> [total, correct]

            with open(merged_path, "w") as fmerge:
                for r in range(world_size):
                    rpath = os.path.join(args.output_dir, f"videommev2_{suffix}_rank{r}.jsonl")
                    if not os.path.exists(rpath):
                        continue
                    with open(rpath) as f:
                        for line in f:
                            fmerge.write(line)
                            item = json.loads(line)
                            total += 1
                            c = 1 if item["correct"] else 0
                            correct += c

                            gt = item.get("group_type", "unknown")
                            if gt not in type_stats:
                                type_stats[gt] = [0, 0]
                            type_stats[gt][0] += 1
                            type_stats[gt][1] += c

                            lv = item.get("level") or "none"
                            if lv not in level_stats:
                                level_stats[lv] = [0, 0]
                            level_stats[lv][0] += 1
                            level_stats[lv][1] += c

            print(f"\n{'='*50}")
            print(f"VideoMME-v2 ({suffix}) | Total: {total}")
            print(f"  Overall Acc: {correct / max(total, 1):.4f} ({correct}/{total})")
            for gt in sorted(type_stats.keys()):
                t, c = type_stats[gt]
                print(f"  {gt:>12s}: {c / max(t, 1):.4f} ({c}/{t})")
            for lv in sorted(level_stats.keys()):
                t, c = level_stats[lv]
                print(f"  Level {lv:>5s}: {c / max(t, 1):.4f} ({c}/{t})")
            print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str,
                        default=None, help="Path to processor (defaults to model_path if not set)")
    parser.add_argument("--output_dir", type=str,
                        default="./logs/videommev2_results")
    parser.add_argument("--with_subtitle", action="store_true",
                        help="Also evaluate with subtitles")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--min_pixels", type=int, default=128 * 2 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=256 * 2 * 32 * 32)
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
