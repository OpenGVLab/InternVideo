import argparse
import json
import os
import re
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==================== 视频路径配置（请修改） ====================
VIDEO_ROOT = {
    "activitynet": "/workspace/activitynet/ANet_320p/val/",
    "charades": os.environ.get("CHARADES_VIDEO_DIR", "/path/to/data/Charades/Charades_v1_480"),
    "qvhighlights": "/workspace/qvhighlight/videos",
}
VIDEO_SUFFIX = {
    "activitynet": ".mp4",
    "charades": ".mp4",
    "qvhighlights": ".mp4",
}
# ===============================================================

DATASET_FILES = {
    "activitynet": os.path.join(os.path.dirname(__file__), "activitynet-timelens.json"),
    "charades": os.path.join(os.path.dirname(__file__), "charades-timelens.json"),
    "qvhighlights": os.path.join(os.path.dirname(__file__), "qvhighlights-timelens.json"),
}

GROUNDING_PROMPT = (
    "Given the video of duration {duration:.1f} seconds, "
    "find the start and end timestamps (in seconds) of the moment that best matches the following description: "
    "\"{query}\"\n"
)


def get_rank_and_world_size():
    """Get rank and world size, support both torchrun and single-process."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def parse_time_span(text):
    """Parse model output to extract start and end time."""
    numbers = re.findall(r"[\d]+\.?\d*", text)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None, None


def build_video_path(video_id, dataset_name):
    root = VIDEO_ROOT[dataset_name]
    suffix = VIDEO_SUFFIX[dataset_name]
    if video_id.endswith(suffix):
        return os.path.join(root, video_id)
    return os.path.join(root, video_id + suffix)


def run_inference(model, processor, video_path, query, duration, fps, min_pixels, max_pixels):
    prompt = GROUNDING_PROMPT.format(duration=duration, query=query)
    mm_processor_kwargs = {
        "min_pixels": min_pixels * 32,
        "max_pixels": max_pixels * 512,
    }
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
        mm_processor_kwargs=mm_processor_kwargs,
    )
    inputs = inputs.to(model.device)
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True,
    #     return_dict=False, return_tensors="pt",
    # )
    # images, videos, video_kwargs = process_vision_info(
    #     messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    # )
    # videos_tensor, vkw = videos[0][0], videos[0][1]
    # inputs = processor(
    #     text=text, images=None, videos=videos_tensor.to(model.device),
    #     do_resize=False, return_tensors="pt", **vkw,
    # )
    # inputs = inputs.to(model.device)

    output = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, output)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return response


def build_query_list(annotations, dataset_name):
    """Flatten annotations into a list of (video_id, query_idx, query, gt_span, duration)."""
    queries = []
    for vid, info in annotations.items():
        duration = info["duration"]
        for qi, (query, gt_span) in enumerate(zip(info["queries"], info["spans"])):
            queries.append((vid, qi, query.strip(), gt_span, duration))
    return queries


def evaluate_dataset(model, processor, dataset_name, args, rank, world_size):
    anno_path = DATASET_FILES[dataset_name]
    with open(anno_path) as f:
        annotations = json.load(f)

    output_path = os.path.join(args.output_dir, f"{dataset_name}_grounding_results_rank{rank}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    # build flat query list and shard by rank
    all_queries = build_query_list(annotations, dataset_name)
    my_queries = all_queries[rank::world_size]

    # resume: load already processed
    done_keys = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                done_keys.add((item["video_id"], item["query_idx"]))

    fout = open(output_path, "a")

    for vid, qi, query, gt_span, duration in tqdm(
        my_queries, desc=f"[{dataset_name}][rank{rank}]", disable=(rank != 0)
    ):
        if (vid, qi) in done_keys:
            continue

        video_path = build_video_path(vid, dataset_name)
        if not os.path.exists(video_path):
            print(f"WARNING [rank{rank}]: video not found: {video_path}")
            continue

        try:
            response = run_inference(
                model, processor, video_path, query, duration,
                args.fps, args.min_pixels, args.max_pixels,
            )
            pred_start, pred_end = parse_time_span(response)
        except Exception as e:
            print(f"ERROR [rank{rank}] {vid} query {qi}: {e}")
            response = ""
            pred_start, pred_end = None, None

        gt_start, gt_end = gt_span[0], gt_span[1]
        iou = 0.0
        if pred_start is not None and pred_end is not None:
            inter_start = max(pred_start, gt_start)
            inter_end = min(pred_end, gt_end)
            inter = max(0, inter_end - inter_start)
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            if union > 0:
                iou = inter / union

        result = {
            "video_id": vid,
            "query_idx": qi,
            "query": query,
            "duration": duration,
            "gt_span": gt_span,
            "pred_span": [pred_start, pred_end],
            "iou": round(iou, 4),
            "response": response,
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()

    # wait for all ranks to finish
    if dist.is_initialized():
        dist.barrier()

    # rank 0 merges results and prints metrics
    if rank == 0:
        merged_path = os.path.join(args.output_dir, f"{dataset_name}_grounding_results.jsonl")
        total, correct_03, correct_05, correct_07, iou_sum = 0, 0, 0, 0, 0.0
        with open(merged_path, "w") as fmerge:
            for r in range(world_size):
                rpath = os.path.join(args.output_dir, f"{dataset_name}_grounding_results_rank{r}.jsonl")
                if not os.path.exists(rpath):
                    continue
                with open(rpath) as f:
                    for line in f:
                        fmerge.write(line)
                        item = json.loads(line)
                        iou = item["iou"]
                        total += 1
                        iou_sum += iou
                        if iou >= 0.3:
                            correct_03 += 1
                        if iou >= 0.5:
                            correct_05 += 1
                        if iou >= 0.7:
                            correct_07 += 1

        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name} | Total: {total}")
        print(f"  mIoU:      {iou_sum / max(total, 1):.4f}")
        print(f"  R@0.3:     {correct_03 / max(total, 1):.4f} ({correct_03}/{total})")
        print(f"  R@0.5:     {correct_05 / max(total, 1):.4f} ({correct_05}/{total})")
        print(f"  R@0.7:     {correct_07 / max(total, 1):.4f} ({correct_07}/{total})")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None, help='Path to processor (defaults to model_path if not set)',
                        help="Processor path, defaults to model_path")
    parser.add_argument("--output_dir", type=str, default="./logs/grounding_results")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["activitynet", "charades", "qvhighlights"],
                        choices=["activitynet", "charades", "qvhighlights"])
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--min_pixels", type=int, default=128 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=512 * 32 * 32)
    args = parser.parse_args()

    # init distributed
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

    for ds in args.datasets:
        if rank == 0:
            print(f"\nEvaluating {ds} ...")
        evaluate_dataset(model, processor, ds, args, rank, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
