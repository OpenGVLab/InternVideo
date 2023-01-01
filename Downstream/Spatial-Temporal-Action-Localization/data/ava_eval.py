import numpy as np
import logging
import tempfile
import os
from pprint import pformat
import csv
import time
from collections import defaultdict


from alphaction.dataset.datasets.evaluation.ava.pascal_evaluation import object_detection_evaluation, standard_fields


def do_ava_evaluation(dataset, predictions, output_folder):
    logging.info("Preparing results for AVA format")
    ava_results = prepare_for_ava_detection(predictions, dataset)
    logging.info("Evaluating predictions")
    # import pdb
    # pdb.set_trace()
    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "result.csv")
        if len(dataset.eval_file_paths) == 0:
            write_csv(ava_results, file_path)
            return
        eval_res = evaluate_predictions_on_ava(
            dataset.eval_file_paths, ava_results, file_path
        )
    logging.info(pformat(eval_res, indent=2))
    if output_folder:
        log_file_path = os.path.join(output_folder, "result.log")
        with open(log_file_path, "a+") as logf:
            logf.write(pformat(eval_res))
    return eval_res, ava_results


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(float(timestamp)))


def decode_image_key(image_key):
    return image_key[:-5], image_key[-4:]


def prepare_for_ava_detection(predictions, dataset):
    # import pdb
    # pdb.set_trace()
    ava_results = {}
    score_thresh = dataset.action_thresh
    TO_REMOVE = 1.0
    for video_id, prediction in enumerate(predictions):
        # import pdb
        # pdb.set_trace()
        video_info = dataset.get_video_info(video_id)
        if len(prediction) == 0:
            continue
        video_width = video_info["width"]
        video_height = video_info["height"]
        prediction = prediction.resize((video_width, video_height))
        prediction = prediction.convert("xyxy")

        boxes = prediction.bbox.numpy()
        boxes[:, [2, 3]] += TO_REMOVE
        boxes[:, [0, 2]] /= video_width
        boxes[:, [1, 3]] /= video_height
        boxes = np.clip(boxes, 0.0, 1.0)

        # No background class.
        scores = prediction.get_field("scores").numpy()
        box_ids, action_ids = np.where(scores >= score_thresh)
        boxes = boxes[box_ids, :]
        scores = scores[box_ids, action_ids]
        action_ids = action_ids + 1

        movie_name = video_info['movie']
        timestamp = video_info['timestamp']

        clip_key = make_image_key(movie_name, timestamp)

        ava_results[clip_key] = {
            "boxes": boxes,
            "scores": scores,
            "action_ids": action_ids
        }
    return ava_results


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
      exclusions_file: Path of file containing a csv of video-id,timestamp.

    Returns:
      A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
      or an empty set if exclusions file is None.
    """
    excluded = set()
    exclusions_file = open(exclusions_file, 'r')
    if exclusions_file:
        reader = csv.reader(exclusions_file)
        for row in reader:
            assert len(row) == 2, "Expected only 2 columns, got: {}".format(row)
            excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
      labelmap_file: Path of file containing a label map protocol buffer.

    Returns:
      labelmap: The label map in the form used by the object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
      class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    labelmap_file = open(labelmap_file, 'r')
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap.append({"id": class_id, "name": name})
            class_ids.add(class_id)
    return labelmap, class_ids


def read_csv(csv_file, class_whitelist=None):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
      csv_file: Path of csv file.
      class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.

    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
      scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    csv_file = open(csv_file, 'r')
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], "Wrong number of columns: " + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue
        score = 1.0
        if len(row) == 8:
            score = float(row[7])
        boxes[image_key].append([y1, x1, y2, x2])
        labels[image_key].append(action_id)
        scores[image_key].append(score)
    print_time("read file " + csv_file.name, start)
    return boxes, labels, scores


def write_csv(ava_results, csv_result_file):
    start = time.time()
    with open(csv_result_file, 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',')
        for clip_key in ava_results:
            movie_name, timestamp = decode_image_key(clip_key)
            cur_result = ava_results[clip_key]
            boxes = cur_result["boxes"]
            scores = cur_result["scores"]
            action_ids = cur_result["action_ids"]
            assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]
            for box, score, action_id in zip(boxes, scores, action_ids):
                box_str = ['{:.5f}'.format(cord) for cord in box]
                score_str = '{:.5f}'.format(score)
                spamwriter.writerow([movie_name, timestamp, ] + box_str + [action_id, score_str])
    print_time("write file " + csv_result_file, start)


def print_time(message, start):
    logging.info("==> %g seconds to %s", time.time() - start, message)


def evaluate_predictions_on_ava(eval_file_paths, ava_results, csv_result_file):
    write_csv(ava_results, csv_result_file)

    groundtruth = eval_file_paths["csv_gt_file"]
    labelmap = eval_file_paths["labelmap_file"]
    exclusions = eval_file_paths["exclusion_file"]

    categories, class_whitelist = read_labelmap(labelmap)
    logging.info("CATEGORIES (%d):\n%s", len(categories),
                 pformat(categories, indent=2))
    excluded_keys = read_exclusions(exclusions)

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)

    # Reads the ground truth dataset.
    boxes, labels, _ = read_csv(groundtruth, class_whitelist)
    start = time.time()
    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(("Found excluded timestamp in ground truth: %s. "
                         "It will be ignored."), image_key)
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key, {
                standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(boxes[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes:
                    np.array(labels[image_key], dtype=int),
                standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(boxes[image_key]), dtype=bool)
            })
    print_time("convert groundtruth", start)

    # Reads detections dataset.
    boxes, labels, scores = read_csv(csv_result_file, class_whitelist)
    start = time.time()
    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(("Found excluded timestamp in detections: %s. "
                         "It will be ignored."), image_key)
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key, {
                standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float)
            })
    print_time("convert detections", start)

    start = time.time()
    metrics = pascal_evaluator.evaluate()
    print_time("run_evaluator", start)
    return metrics
