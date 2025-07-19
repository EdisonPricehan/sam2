# Segment Anything Model 2 (SAM2) - Online Segmentation
This project extends the original [Segment Anything Model 2 (SAM2) Video Predictor](https://github.com/facebookresearch/sam2) to support true online segmentation of a single object from live camera or video files, with interactive initialization and bounded memory usage.

## What’s Changed

- **Interactive Initialization**: Added PointSelector to collect foreground/background clicks on the first frame via OpenCV.

- **Streaming Loop**: Replaced propagate_in_video batch predictor with infer_online_stream, processing each incoming frame in real time without preloading the full sequence.

- **Prune State**: Introduced prune_state to trim growing dictionaries (cached_features, frames_tracked_per_obj, prompt inputs, temp outputs, non‑conditioned masks) down to the latest N frames—preventing OOM on long streams.

- **Sliding‑Window Buffer**: Retained global frame indexing while bounding GPU usage by pruning only unreferenced caches.

- **Performance Overlay**: Added put_stats_on_frame to display FPS, latency, and memory usage on each output frame.

- **Offline Mode**: Included infer_offline_video to run on pre‑extracted frames or video files using the same interactive prompt.

- **Save Output**: Configurable save_video flag writes results to MP4 with correct dimensions.


### Major Change
Major changes are made in `tryit.py`, which is the main script for running the online segmentation. 

### Minor Changes
Changes are also made in `sam2_video_predictor.py`, where a miscall of `_clear_non_cond_mem_around_input` was fixed, and frame index was tweaked in function `_get_image_feature` since we only keep the latest N frames in memory.
Besides, the `image_size` parameter in config files is set to be the same as the `img_height` and `img_width` in the `tryit.py` script, so that the model can be loaded correctly.

## Usage
Edit `tryit.py` configuration near the bottom:
```python
# Model & checkpoint
checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg  = "./configs/sam2.1/sam2.1_hiera_s.yaml"

# Inference mode flags
infer_offline = False    # `True` to run on extracted frames; `False` for live stream or file
infer_webcam  = False    # `True` to open webcam; ignored if `infer_offline=True`
save_video    = True     # `True` to write out MP4 to `videos/results/`
keep_N_frames = 5        # Number of frames to keep in the state to limit memory usage

# Paths for offline mode
video_path = "./notebooks/videos/your_video.mp4"
frames_dir = "./notebooks/videos/your_frames_dir"
points_file = "./notebooks/videos/your_prompts.json"

# Image size (must match model config)
img_height, img_width = 512, 512
```

Once you run the script, it will pop up a window showing the first frame (either from webcam of a video file). 
You can left click on the foreground (and optionally right click background) to initialize the segmentation, then press `q` to start the online segmentation.
The script will then process each frame in real time, displaying the segmentation results and performance statistics.

## Example Online Inference Video
Stats are based on a NVIDIA RTX 4060 GPU with 8GB memory, and the model is SAM2.1 Hierarchical Small.

![Wabash River Video](./notebooks/videos/results/wabash_upstream_fastforward_60x_512x512_online.mp4)

![Webcam Stream](./notebooks/videos/results/webcam_20250718_170455.mp4)

## Issues
Still, the GPU memory usage will grow over time, just at a very slower rate than the original SAM2 Video Predictor.
If you still encounter OOM issues and found a way to fix it, please open a PR.