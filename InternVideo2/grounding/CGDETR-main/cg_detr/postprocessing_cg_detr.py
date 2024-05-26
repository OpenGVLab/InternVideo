import pprint
import numpy as np
import torch
from utils.basic_utils import load_jsonl
from standalone_eval.eval import eval_submission
from tqdm import tqdm


class PostProcessorDETR:
    def __init__(self, clip_length=2, min_ts_val=0, max_ts_val=150,
                 min_w_l=2, max_w_l=70, move_window_method="center",
                 process_func_names=("clip_window_l", "clip_ts", "round_multiple")):
        self.clip_length = clip_length
        self.min_ts_val = min_ts_val
        self.max_ts_val = max_ts_val
        self.min_w_l = min_w_l
        self.max_w_l = max_w_l
        self.move_window_method = move_window_method
        self.process_func_names = process_func_names
        self.name2func = dict(
            clip_ts=self.clip_min_max_timestamps,
            round_multiple=self.round_to_multiple_clip_lengths,
            clip_window_l=self.clip_window_lengths
        )

    def __call__(self, lines):
        processed_lines = []
        for line in tqdm(lines, desc=f"convert to multiples of clip_length={self.clip_length}"):
            windows_and_scores = torch.tensor(line["pred_relevant_windows"])
            windows = windows_and_scores[:, :2]
            for func_name in self.process_func_names:
                windows = self.name2func[func_name](windows)
            line["pred_relevant_windows"] = torch.cat(
                [windows, windows_and_scores[:, 2:3]], dim=1).tolist()
            line["pred_relevant_windows"] = [e[:2] + [float(f"{e[2]:.4f}")] for e in line["pred_relevant_windows"]]
            processed_lines.append(line)
        return processed_lines

    def clip_min_max_timestamps(self, windows):
        """
        windows: (#windows, 2)  torch.Tensor
        ensure timestamps for all windows is within [min_val, max_val], clip is out of boundaries.
        """
        return torch.clamp(windows, min=self.min_ts_val, max=self.max_ts_val)

    def round_to_multiple_clip_lengths(self, windows):
        """
        windows: (#windows, 2)  torch.Tensor
        ensure the final window timestamps are multiples of `clip_length`
        """
        return torch.round(windows / self.clip_length) * self.clip_length

    def clip_window_lengths(self, windows):
        """
        windows: (#windows, 2)  np.ndarray
        ensure the final window duration are within [self.min_w_l, self.max_w_l]
        """
        window_lengths = windows[:, 1] - windows[:, 0]
        small_rows = window_lengths < self.min_w_l
        if torch.sum(small_rows) > 0:
            windows = self.move_windows(
                windows, small_rows, self.min_w_l, move_method=self.move_window_method)
        large_rows = window_lengths > self.max_w_l
        if torch.sum(large_rows) > 0:
            windows = self.move_windows(
                windows, large_rows, self.max_w_l, move_method=self.move_window_method)
        return windows

    @classmethod
    def move_windows(cls, windows, row_selector, new_length, move_method="left"):
        """
        Args:
            windows:
            row_selector:
            new_length:
            move_method: str,
                left: keep left unchanged
                center: keep center unchanged
                right: keep right unchanged

        Returns:

        """
        # import ipdb;
        # ipdb.set_trace()
        if move_method == "left":
            windows[row_selector, 1] = windows[row_selector, 0] + new_length
        elif move_method == "right":
            windows[row_selector, 0] = windows[row_selector, 1] - new_length
        elif move_method == "center":
            center = (windows[row_selector, 1] + windows[row_selector, 0]) / 2.
            windows[row_selector, 0] = center - new_length / 2.
            windows[row_selector, 1] = center + new_length / 2.
        return windows

