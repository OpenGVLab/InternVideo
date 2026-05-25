import argparse
import json


DEFAULT_RESULT_PATH = "grounding_results/charades_grounding_results.jsonl"


def load_unique_results(result_path):
    results = []
    seen = set()

    with open(result_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["video_id"], item["query_idx"])
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", nargs="?", default=DEFAULT_RESULT_PATH)
    args = parser.parse_args()

    results = load_unique_results(args.result_path)
    total = len(results)
    iou_sum = sum(item["iou"] for item in results)
    r03 = sum(1 for item in results if item["iou"] >= 0.3)
    r05 = sum(1 for item in results if item["iou"] >= 0.5)
    r07 = sum(1 for item in results if item["iou"] >= 0.7)

    print(f"{'=' * 50}")
    print(f"Result file: {args.result_path}")
    print(f"Total: {total}")
    print(f"mIoU:  {iou_sum / max(total, 1):.4f}")
    print(f"R@0.3: {r03 / max(total, 1):.4f} ({r03}/{total})")
    print(f"R@0.5: {r05 / max(total, 1):.4f} ({r05}/{total})")
    print(f"R@0.7: {r07 / max(total, 1):.4f} ({r07}/{total})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
