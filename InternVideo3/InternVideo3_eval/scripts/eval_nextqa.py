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
DATA_ROOT = os.environ.get("DATA_DIR", "/path/to/data/nextqa")
CSV_PATH = os.path.join(DATA_ROOT, "val.csv")
# =====================================================

MCQ_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "{question}\n{options}\n"
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
    m = re.match(r"^([A-E])", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"(?:answer is|answer:)?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return text.strip()[0].upper() if text.strip() else ""


def find_video_path(video_id):
    """Find video file in subdirectories."""
    for subdir in os.listdir(DATA_ROOT):
        subdir_path = os.path.join(DATA_ROOT, subdir)
        if not os.path.isdir(subdir_path):
            continue
        video_path = os.path.join(subdir_path, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def build_video_index():
    """Build a mapping from video_id to file path."""
    index = {}
    for subdir in os.listdir(DATA_ROOT):
        subdir_path = os.path.join(DATA_ROOT, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.endswith(".mp4"):
                vid = fname[:-4]
                index[vid] = os.path.join(subdir_path, fname)
    return index


def build_samples(csv_path):
    """Load CSV and return list of sample dicts."""
    df = pd.read_csv(csv_path)
    samples = []
    option_letters = ["A", "B", "C", "D", "E"]
    for _, row in df.iterrows():
        options = []
        for i, letter in enumerate(option_letters):
            col = f"a{i}"
            if col in row and pd.notna(row[col]):
                options.append(f"{letter}. {row[col]}")
        options_str = "\n".join(options)

        # answer is the index (0-4), convert to letter
        answer_letter = option_letters[int(row["answer"])]

        samples.append({
            "video_id": str(row["video"]),
            "question_id": int(row["qid"]),
            "question": row["question"],
            "options_str": options_str,
            "answer": answer_letter,
            "type": row["type"],
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
    samples = build_samples(CSV_PATH)
    my_samples = samples[rank::world_size]

    # Build video index once
    if rank == 0:
        print("Building video index...")
    video_index = build_video_index()
    if rank == 0:
        print(f"Found {len(video_index)} videos in index.")

    output_path = os.path.join(args.output_dir, f"nextqa_rank{rank}.jsonl")
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
        my_samples, desc=f"[NExT-QA][rank{rank}]", disable=(rank != 0)
    ):
        if sample["question_id"] in done_keys:
            continue

        video_path = video_index.get(sample["video_id"])
        if video_path is None or not os.path.exists(video_path):
            print(f"WARNING [rank{rank}]: video not found: {sample['video_id']}")
            continue

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
            print(f"ERROR [rank{rank}] qid={sample['question_id']}: {e}")
            response = ""
            pred = ""

        result = {
            "video_id": sample["video_id"],
            "question_id": sample["question_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "pred": pred,
            "correct": pred == sample["answer"],
            "type": sample["type"],
            "response": response,
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()

    if dist.is_initialized():
        dist.barrier()

    # rank 0 merges and reports
    if rank == 0:
        merged_path = os.path.join(args.output_dir, "nextqa_results.jsonl")
        total, correct = 0, 0
        type_stats = {}  # type -> [total, correct]

        with open(merged_path, "w") as fmerge:
            for r in range(world_size):
                rpath = os.path.join(args.output_dir, f"nextqa_rank{r}.jsonl")
                if not os.path.exists(rpath):
                    continue
                with open(rpath) as f:
                    for line in f:
                        fmerge.write(line)
                        item = json.loads(line)
                        total += 1
                        c = 1 if item["correct"] else 0
                        correct += c

                        t = item.get("type", "unknown")
                        if t not in type_stats:
                            type_stats[t] = [0, 0]
                        type_stats[t][0] += 1
                        type_stats[t][1] += c

        print(f"\n{'='*50}")
        print(f"NExT-QA | Total: {total}")
        print(f"  Overall Acc: {correct / max(total, 1):.4f} ({correct}/{total})")
        for t in sorted(type_stats.keys()):
            tot, cor = type_stats[t]
            print(f"  {t:>12s}: {cor / max(tot, 1):.4f} ({cor}/{tot})")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str,
                        default=None, help="Path to processor (defaults to model_path if not set)")
    parser.add_argument("--output_dir", type=str,
                        default="./logs/nextqa_results")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--min_pixels", type=int, default=256 * 2 * 32 * 32)
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
