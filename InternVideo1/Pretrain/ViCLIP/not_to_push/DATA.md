### Download

#### Pre-Training 

- CC12M images, https://github.com/google-research-datasets/conceptual-12m
- CC3M images, https://github.com/google-research-datasets/conceptual-captions
- SBU images, https://www.cs.rice.edu/~vo9/sbucaptions/
- COCO images, https://cocodataset.org/#download
- VG images, https://visualgenome.org/api/v0/api_home.html
- WebVid videos, https://github.com/m-bain/webvid

For datasets that only provide urls, you may use [img2dataset](https://github.com/rom1504/img2dataset) to speed up downloading.

#### Downstream

- MSRVTT videos, https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
- ActivityNet videos, http://activity-net.org/download.html
- DiDeMo videos, https://github.com/LisaAnne/LocalizingMoments
- SSv2 videos, https://developer.qualcomm.com/software/ai-datasets/something-something
- Flickr30K images, http://shannon.cs.illinois.edu/DenotationGraph/


#### Compressing Videos and Images
We preprocess videos/images to lower FPS and dimension to reduce storage and to improve data loading. For videos, you may use
```bash
fps=2
size=224
file_type=video
input_root=/path/to/webvid_videos
input_file_list_path=/path/to/webvid_video_names.txt
# you may use `ls -U ${input_root} > ${input_file_list_path}` to efficiently generate the file above.
output_root=/path/to/processed_webvid_videos
python preprocess/compress.py \
--input_root=${input_root} --output_root=${output_root} \
--input_file_list_path=${input_file_list_path} \
--fps=${fps} --size=${size} --file_type=${file_type} --num_workers 24 
```
Note that the audio is also removed from the video files, you need edit the file [preprocess/compress.py](preprocess/compress.py) to keep it. For images, you may use
```bash
size=224
file_type=images
input_root=/path/to/cc3m_images
input_file_list_path=/path/to/cc3m_image_names.txt
# you may use `ls -U ${input_root} > ${input_file_list_path}` to efficiently generate the file above.
output_root=/path/to/processed_cc3m_images
python preprocess/compress.py \
--input_root=${input_root} --output_root=${output_root} \
--input_file_list_path=${input_file_list_path} \
--size=${size} --file_type=${file_type} --num_workers 24 
```

