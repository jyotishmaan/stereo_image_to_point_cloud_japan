# Stereo Video to Point Clouds

## Getting Started

1. clone the repository using `git clone https://github.com/wuxiaohua1011/stereo_image_to_point_cloud_japan.git`
2. Setup an identical conda environment using
   `conda create --name myenv --file spec-file.txt`
3. Activate the environment using `conda activate myenv`
4. install required modules `pip install tqdm opencv-python`
5. Download the data video file `mkdir data/raw_data && python download_google_drive.py 1DZvYQMRoPbrmJ67q4kNZ1duFWk5oHnV_ ./data/raw_data/both_eye_0304_1.mp4`
6. To gather data from video, use `python video_to_frames.py`, please note that you can modify the variable N to extend or shorten the frames collected
7. If you already have a series of stereo images, simply verify that the file path is correct in `stereo_match_multiple.py`, and use the command `python stereo_match_multiple.py`
