# Stereo Video to Point Clouds

## Getting Started

1. clone the repository using `git clone https://github.com/wuxiaohua1011/stereo_image_to_point_cloud_japan.git`
2. Setup an identical conda environment using
   `conda create --name myenv --file spec-file.txt`
3. Activate the environment using `conda activate myenv`
4. install required modules `pip install tqdm opencv-python`
5. Download the data video file `mkdir data/raw_data && python download_google_drive.py 1DZvYQMRoPbrmJ67q4kNZ1duFWk5oHnV_ ./data/raw_data/both_eye_0304_1.mp4`
6. run `python together.py`, you should see a windows pop up, press `s` to draw a bounding box, and then press `space` or `enter` to confirm the bounding box, then you should see both the position and the disparity map showing.

Feel free to disable showing disparity map by passing in False on line 102 for calculating disparity map

#### Sample Image of the program
Sample Program Run Without Selection
![Sample Program Run Without Selection](./paper/images/sample_program_run_without_selection.png)

Sample Program Run With Selection
![Sample Program Run With Selection](./paper/images/sample_program_run_with_selection.png)
