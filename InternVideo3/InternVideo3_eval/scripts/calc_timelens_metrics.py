import json
import os
import sys
from collections import defaultdict

result_dir = sys.argv[1] if len(sys.argv) > 1 else "timelens_results"

# collect all rank files
results = []
for fname in sorted(os.listdir(result_dir)):
    if fname.startswith("timelens_results") and fname.endswith(".jsonl"):
        with open(os.path.join(result_dir, fname)) as f:
            for line in f:
                results.append(json.loads(line))

# deduplicate by (line_idx, ev_idx)
seen = set()
unique = []
for r in results:
    key = (r["line_idx"], r["ev_idx"])
    if key not in seen:
        seen.add(key)
        unique.append(r)

# overall metrics
total = len(unique)
iou_sum = sum(r["iou"] for r in unique)
r03 = sum(1 for r in unique if r["iou"] >= 0.3)
r05 = sum(1 for r in unique if r["iou"] >= 0.5)
r07 = sum(1 for r in unique if r["iou"] >= 0.7)

print(f"{'='*50}")
print(f"TimeLens Overall | Total: {total}")
print(f"  mIoU:   {iou_sum / max(total, 1):.4f}")
print(f"  R@0.3:  {r03 / max(total, 1):.4f} ({r03}/{total})")
print(f"  R@0.5:  {r05 / max(total, 1):.4f} ({r05}/{total})")
print(f"  R@0.7:  {r07 / max(total, 1):.4f} ({r07}/{total})")
print(f"{'='*50}")

# per-source breakdown
by_source = defaultdict(list)
for r in unique:
    by_source[r.get("source", "unknown")].append(r)

for source in sorted(by_source):
    items = by_source[source]
    n = len(items)
    s = sum(r["iou"] for r in items)
    c3 = sum(1 for r in items if r["iou"] >= 0.3)
    c5 = sum(1 for r in items if r["iou"] >= 0.5)
    c7 = sum(1 for r in items if r["iou"] >= 0.7)
    print(f"\n  [{source}] Total: {n}")
    print(f"    mIoU:  {s / max(n, 1):.4f}")
    print(f"    R@0.3: {c3 / max(n, 1):.4f} ({c3}/{n})")
    print(f"    R@0.5: {c5 / max(n, 1):.4f} ({c5}/{n})")
    print(f"    R@0.7: {c7 / max(n, 1):.4f} ({c7}/{n})")
