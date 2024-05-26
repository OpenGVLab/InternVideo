"""
Find windows from a video with clip_ids.

A window is defined by a [start_clip_idx, end_clip_idx] pair:
For example, assuming clip_len = 2 seconds
[0, 0] meaning a single clip window [0, 2] (seconds)
[10, 19] meaning a 9 clip window [20, 40] (seconds)

"""


def convert_clip_ids_to_windows(clip_ids):
    """ Inverse function of convert_windows_to_clip_ids
    Args:
        clip_ids: list(int), each is a index of a clip, starting from 0

    Returns:
        list(list(int)), each sublist contains two integers which are clip indices.
            [10, 19] meaning a 9 clip window [20, 40] (seconds), if each clip is 2 seconds.

    >>> test_clip_ids = [56, 57, 58, 59, 60, 61, 62] + [64, ] + [67, 68, 69, 70, 71]
    >>> convert_clip_ids_to_windows(test_clip_ids)
    [[56, 62], [64, 64], [67, 71]]
    """
    windows = []
    _window = [clip_ids[0], None]
    last_clip_id = clip_ids[0]
    for clip_id in clip_ids:
        if clip_id - last_clip_id > 1:  # find gap
            _window[1] = last_clip_id
            windows.append(_window)
            _window = [clip_id, None]
        last_clip_id = clip_id
    _window[1] = last_clip_id
    windows.append(_window)
    return windows


def convert_windows_to_clip_ids(windows):
    """ Inverse function of convert_clip_ids_to_windows
    Args:
        windows: list(list(int)), each sublist contains two integers which are clip indices.
            [10, 11] meaning a 9 clip window [20, 40] (seconds), if each clip is 2 seconds.

    Returns:
        clip_ids: list(int)
        
    >>> test_windows =[[56, 62], [64, 64], [67, 71]]
    >>> convert_windows_to_clip_ids(test_windows)
     [56, 57, 58, 59, 60, 61, 62] + [64, ] + [67, 68, 69, 70, 71]
    """
    clip_ids = []
    for w in windows:
        clip_ids += list(range(w[0], w[1]+1))
    return clip_ids


def convert_clip_window_to_seconds(window, clip_len=2):
    return [window[0] * clip_len, (window[1] + 1) * clip_len]
