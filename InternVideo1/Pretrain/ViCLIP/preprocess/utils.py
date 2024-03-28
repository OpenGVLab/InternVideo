import json
import subprocess


def get_video_duration(filename):

    result = subprocess.check_output(
        f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"', shell=True
    ).decode()
    fields = json.loads(result)["streams"][0]

    duration = float(fields["duration"])
    return duration

if __name__ == "__main__":
    import os
    fp = os.path.join(os.environ["SL_DATA_DIR"], "videos_images/webvid_10m_2fps_224/22920757.mp4")
    print(get_video_duration(fp))
