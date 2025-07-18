import torch
import cv2
import os
import json
import time
from glob import glob
from collections import OrderedDict, deque
from typing import Optional, Tuple, List
from datetime import datetime
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class PointSelector:
    """
    A class to handle interactive point selection on an image using OpenCV.
    Clicking left selects a point as foreground, right selects it as background.
    """
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image
        self.points = []
        self.labels = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click = foreground
            self.points.append([x, y])
            self.labels.append(1)  # Positive click
            # Draw point on image
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.image)

        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click = background
            self.points.append([x, y])
            self.labels.append(0)  # Negative click
            # Draw point on image
            cv2.circle(self.image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(self.window_name, self.image)


def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Extract a specific frame from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_idx (int): Index of the frame to extract.

    Returns:
        Optional[numpy.ndarray]: The extracted frame as a numpy array, or None if extraction failed.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_interactive_points(frame: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    """
    Get interactive points from the user on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to select points on, [height, width, 3].

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - points: List of selected points as [x, y] coordinates.
            - labels: List of labels (1 for foreground, 0 for background).
    """
    window_name = 'Select Points'
    frame_copy = frame.copy()

    selector = PointSelector(window_name, frame_copy)
    cv2.imshow(window_name, frame_copy)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    print("Left click for foreground, Right click for background")
    print("Press 'q' when done selecting points")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return selector.points, selector.labels


def display_results(frame: np.ndarray, masks_logits: torch.Tensor, alpha=0.5, fig=None, ax=None):
    """
    Display the frame and masks in a matplotlib figure.
    Args:
        frame (numpy.ndarray): The video frame to display, [height, width, 3].
        masks_logits (torch.Tensor): The masks logits from the model, [num_objects, 1, height, width].
        alpha (float): Transparency for the mask overlay.
        fig (matplotlib.figure.Figure): Optional figure to use for display.
        ax (matplotlib.axes.Axes): Optional axes to use for display.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, numpy.ndarray]: The figure, axes, and colored mask.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.show(block=False)  # Show but don't block

    # Clear previous content
    ax.clear()

    # Display frame and masks
    ax.imshow(frame[..., ::-1])  # BGR to RGB
    colored_mask = np.zeros_like(frame)
    for mask in masks_logits:
        colored_mask = np.zeros_like(frame)
        mask_bool = mask.cpu().numpy() > 0
        colored_mask[mask_bool[0]] = [0, 255, 0]  # Green mask
        ax.imshow(colored_mask, alpha=alpha)

    ax.axis('off')
    fig.canvas.draw()
    plt.pause(0.001)  # Small pause to update the plot

    return fig, ax, colored_mask

def overlay_masks(frame: np.ndarray, masks_logits: torch.Tensor) -> np.ndarray:
    """
    Overlay binary masks on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to overlay masks on, [height, width, 3].
        masks_logits (torch.Tensor): The masks logits from the model, [num_objects, 1, height, width].

    Returns:
        numpy.ndarray: The frame with masks overlaid, [height, width, 3].
    """
    result_frame = frame.copy()

    for mask in masks_logits:
        # Convert logits to binary mask efficiently
        mask_bool = (mask.cpu().numpy() > 0).squeeze()

        # Resize mask to frame size
        mask_resized = cv2.resize(
            mask_bool.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST  # Faster interpolation
        )

        # Create colored mask overlay
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_resized > 0] = [0, 255, 0]  # Green mask

        # Overlay mask on frame
        result_frame = cv2.addWeighted(result_frame, 1.0, colored_mask, 0.5, 0)

    return result_frame


def put_stats_on_frame(frame: np.ndarray, frame_idx: int, stats: dict) -> np.ndarray:
    """
    Display performance statistics on the video frame.

    Args:
        frame (numpy.ndarray): The video frame to display stats on.
        frame_idx (int): The current frame index.
        stats (dict): A dictionary containing performance statistics.

    Returns:
        numpy.ndarray: The frame with statistics displayed.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw performance metrics
    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                font, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Time: {stats['avg_processing_time'] * 1000:.1f}ms", (10, 60),
                font, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Memory: {stats['memory_usage_mb']:.1f}MB", (10, 90),
                font, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 120),
                font, 0.7, (0, 0, 255), 2)

    return frame


def load_frames_from_dir(frames_dir) -> List[str]:
    """
    Load frame paths sorted by frame number.

    Args:
        frames_dir (str): Directory containing frame images.

    Returns:
        List[str]: Sorted list of frame image paths.
    """
    frame_paths = glob(os.path.join(frames_dir, "*.jpg"))
    # frame_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    frame_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    return frame_paths


def extract_frames_from_video(video_path: str, frames_dir: str) -> List[str]:
    """
    Extract frames from a video file and save them to a directory.

    Args:
        video_path (str): Path to the video file.
        frames_dir (str): Directory to save extracted frames.

    Returns:
        List[str]: List of paths to the extracted frame images.
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()

    print(f"Extracted {frame_idx} frames to {frames_dir}")
    return frame_paths


def save_points(points: List[List[int]], labels: List[int], save_path: str):
    """
    Save prompt points and labels to JSON.

    Args:
        points (List[List[int]]): List of points as [x, y] coordinates.
        labels (List[int]): List of labels (1 for foreground, 0 for background).
        save_path (str): Path to save the JSON file.
    """
    data = {
        'points': points,
        'labels': labels
    }
    with open(save_path, 'w') as f:
        json.dump(data, f)


def load_points(save_path: str) -> Tuple[List[List[int]], List[int]]:
    """
    Load prompt points and labels from JSON.

    Args:
        save_path (str): Path to the JSON file containing points and labels.

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple containing:
            - points: List of points as [x, y] coordinates.
            - labels: List of labels (1 for foreground, 0 for background).
    """
    with open(save_path, 'r') as f:
        data = json.load(f)
    return data['points'], data['labels']


def infer_offline_video():
    # Initialize predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    print("Predictor initialized.")

    with torch.inference_mode():
        print('Initializing state...')
        # state = predictor.init_state(video_path=video_path)  # Use video path if you want to process the video directly
        state = predictor.init_state(video_path=frames_dir)  # Use extracted frames directory
        print('State inited.')

        print(f'{state["images"].shape=}')

        # Get initial frame results
        frame_idx, object_ids, masks_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,  # The frame index where we choose the points and labels
            obj_id=1,  # A unique integer ID for the object
            points=points,  # Pass points directly
            labels=labels,  # Pass labels directly
        )
        print(f"Initial frame masks shape: {masks_logits.shape}, frame_idx: {frame_idx}, obj_ids: {object_ids}")
        # display_results(first_frame, masks_logits)

        # Propagate through video
        fig = None
        ax = None
        print("Processing video frames...")
        for i, (frame_idx, object_ids, masks_logits) in enumerate(predictor.propagate_in_video(state)):
            if i >= len(frame_paths):
                break

            frame = cv2.imread(frame_paths[i])
            if frame is None:
                continue

            fig, ax, colored_mask = display_results(frame, masks_logits, fig=fig, ax=ax)

            if save_video:
                # Convert colored mask to BGR for saving
                colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                output_frame = cv2.addWeighted(frame, 1.0, colored_mask_bgr, 0.5, 0)
                video_writer.write(output_frame)

            print(f"Processed frame {i + 1}/{len(frame_paths)}")
            torch.cuda.empty_cache()

        if save_video:
            video_writer.release()
            print(f"Output video saved to {output_video_path}")


def preprocess_frame(frame: np.ndarray, img_h: int, img_w: int) -> torch.Tensor:
    """
    Convert a video frame to a tensor suitable for model input.
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    frame_resized = cv2.resize(frame_rgb, (img_h, img_w))

    # Convert to tensor and normalize efficiently
    frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
    frame_tensor = frame_tensor.to(device)  # <-- Move to device BEFORE normalization

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    frame_tensor = (frame_tensor - mean) / std

    return frame_tensor


def prune_state(state, max_frames: int = 10):
    """
    In-place pruning of SAM2 inference_state to bound GPU memory usage
    while preserving all conditioning-frame data.

    Args:
        state (dict): The inference state dictionary containing all necessary data.
        max_frames (int): The maximum number of frames to keep in the state.
    """
    num_frames = state["num_frames"]
    current_frame = num_frames - 1
    prune_before = current_frame - max_frames

    # Update the number of pruned frames
    if num_frames > max_frames:
        state["pruned_frames_num"] = num_frames - max_frames

    # 1. Keep only the last N raw frames
    state["images"] = state["images"][-max_frames:]

    # 2. Prune cached image features older than the window
    for idx in list(state["cached_features"]):
        if idx < prune_before:
            del state["cached_features"][idx]

    # 3. Drop old frame metadata
    for obj_meta in state["frames_tracked_per_obj"].values():
        for fidx in list(obj_meta):
            if fidx < prune_before:
                obj_meta.pop(fidx)

    # 4. Clear any stray temp outputs (usually empty after preflight)
    # for obj_temp in state["temp_output_dict_per_obj"].values():
    #     obj_temp["cond_frame_outputs"].clear()
    #     obj_temp["non_cond_frame_outputs"].clear()
    for temp_out in state.get("temp_output_dict_per_obj", {}).values():
        for store in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for fidx in list(temp_out.get(store, {})):
                if fidx < prune_before:
                    temp_out[store].pop(fidx, None)

    # Prune prompt inputs
    for pts in state.get("point_inputs_per_obj", {}).values():
        for fidx in list(pts):
            if fidx < prune_before:
                pts.pop(fidx, None)
    for ms in state.get("mask_inputs_per_obj", {}).values():
        for fidx in list(ms):
            if fidx < prune_before:
                ms.pop(fidx, None)

    torch.cuda.empty_cache()

    return state


@torch.inference_mode()
def infer_online_stream():
    # 1. Predictor & Capture
    predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    predictor.clear_non_cond_mem_around_input = True
    print('Predictor initialized.')

    # Open video capture (webcam or video file)
    cap = cv2.VideoCapture(0)  # web camera or video file path

    # 2. First frame & prompts
    ret, frame = cap.read()
    if not ret:
        print("No stream available.")
        return

    # Resize the frame to the desired size
    frame = cv2.resize(frame, (img_height, img_width), interpolation=cv2.INTER_NEAREST)

    # Get interactive points from the first frame
    points, labels = get_interactive_points(frame)

    # 3. Build minimal state
    first_tensor = preprocess_frame(frame, img_height, img_width)  # (3, height, width)

    state = {
        "device": device,
        "storage_device": device,
        "pruned_frames_num": 0,
        "constants": {},
        "video_height": img_height,
        "video_width": img_width,
        "images": first_tensor.unsqueeze(0),  # Start with the first frame, (1, 3, height, width)
        "num_frames": 1,  # Start with one frame
        "cached_features": {},
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
    }
    predictor._get_image_feature(inference_state=state, frame_idx=0, batch_size=1)

    # 4. Initial mask & memory
    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,  # Unique ID for the object
        points=points,
        labels=labels,
    )
    print(f'Points are added.')
    # print(f'{predictor.image_size=} {state["video_height"]=} {state["video_width"]=}')

    try:
        # 5. Stream loop
        fig, ax = None, None  # For persistent display
        processing_times = deque(maxlen=50)  # For performance stats
        while True:
            # Read a new frame from the video capture
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the desired size
            frame = cv2.resize(frame, (img_height, img_width), interpolation=cv2.INTER_NEAREST)

            tensor = preprocess_frame(frame, img_height, img_width)  # (3, height, width)

            # Concat the new frame tensor to the images tensor
            state["images"] = torch.cat((state["images"], tensor.unsqueeze(0)), dim=0)  # (num_frames, 3, height, width)

            state["num_frames"] += 1
            frame_id = state["num_frames"] - 1

            # Prune the state to keep only the last `max_frames` frames
            state = prune_state(state, max_frames=5)  # Keep the last N frames in the state

            # print(f'output_dict_per_obj: {state["output_dict_per_obj"][0]=}')
            # print(f'{frame_id=} {state["num_frames"]=} {state["obj_ids"]=}')

            start_time = time.time()

            predictor.propagate_in_video_preflight(state)

            obj_output_dict = state["output_dict_per_obj"][0]  # Interested in the first and only object

            current_out, mask_logits = predictor._run_single_frame_inference(
                inference_state=state,
                output_dict=obj_output_dict,
                frame_idx=frame_id,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )

            # All frames other than the first one (where object of interest is selected) are non-conditional frames
            obj_output_dict["non_cond_frame_outputs"][frame_id] = current_out

            # Get the mask logits in the original resolution for the current frame
            _, mask_logits = predictor._get_orig_video_res_output(
                state, mask_logits
            )

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            stats = get_performance_stats(processing_times)

            # fig, ax, colored_mask = display_results(frame=frame, masks_logits=mask_logits, alpha=0.5, fig=fig, ax=ax)

            # Embed performance stats on the mask-overlaid frame
            overlaid_frame = overlay_masks(frame=frame, masks_logits=mask_logits)
            overlaid_frame_with_stats = put_stats_on_frame(
                frame=overlaid_frame,
                frame_idx=frame_id,
                stats=stats
            )
            # print(f'{overlaid_frame_with_stats.shape=}')

            # Display the frame with masks and stats
            cv2.imshow("Video Stream with Masks", overlaid_frame_with_stats)
            cv2.waitKey(1)  # Allow OpenCV to process the window events

            # Save to video if enabled
            if save_video:
                # print(f'Writing frame {frame_id} to video...')
                video_writer.write(overlaid_frame_with_stats)

    except Exception as e:
        print(f'Error during inference: {e}')
    finally:
        # 6. Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if save_video:
            video_writer.release()


def get_performance_stats(processing_times: deque[float]) -> dict:
    """
    Get performance statistics for monitoring.

    Args:
        processing_times (List[float]): List of processing times for each frame.

    Returns:
        dict: A dictionary containing average processing time, FPS, and memory usage.
    """
    stats = {
        'avg_processing_time': 0.0,
        'fps': 0.0,
        'memory_usage_mb': 0.0,
    }

    if processing_times:
        stats['avg_processing_time'] = sum(processing_times) / len(processing_times)
        stats['fps'] = 1.0 / stats['avg_processing_time'] if stats['avg_processing_time'] > 0 else 0.0

    # Get memory usage
    if device.type == "cuda":
        stats['memory_usage_mb'] = torch.cuda.memory_allocated() / 1024 / 1024

    return stats


if __name__ == '__main__':
    # Model setup
    checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    # checkpoint = "./checkpoints/sam2.1_hiera_small.pt"

    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # Video path setup
    # video_path = './notebooks/videos/wabash_upstream_640x480_1fps.mp4'
    # video_path = './notebooks/videos/wabash_upstream_true1fps.mp4'
    # video_path = './notebooks/videos/wabash_upstream_fastforward_60x.mp4'
    # video_path = './notebooks/videos/wabash_upstream_fastforward_60x_128x128.mp4'
    video_path = './notebooks/videos/wabash_upstream_fastforward_60x_512x512.mp4'
    # video_path = './notebooks/videos/bedroom.mp4'

    # Video frames directory setup
    # frames_dir = "./notebooks/videos/wabash_upstream"
    # frames_dir = "./notebooks/videos/wabash_upstream_128x128"
    frames_dir = "./notebooks/videos/wabash_upstream_512x512"
    # frames_dir = "./notebooks/videos/bedroom"

    # Points file setup
    # points_file = "notebooks/videos/prompt_points_bedroom.json"
    # points_file = "notebooks/videos/prompt_points_wabash_upstream.json"
    # points_file = "notebooks/videos/prompt_points_wabash_upstream_128x128.json"
    points_file = "notebooks/videos/prompt_points_wabash_upstream_512x512.json"

    # Create results directory
    results_dir = "./notebooks/videos/results"
    os.makedirs(results_dir, exist_ok=True)

    # Configurable variables
    save_video = True  # Set to True if you want to save the output video
    infer_offline = False  # True for offline inference with pre-extracted frames, False for online stream inference
    # img_height, img_width = 1024, 1024  # Has to be aligned with the param "image_size" in the model config yaml file
    img_height, img_width = 512, 512  # Has to be aligned with the param "image_size" in the model config yaml file

    # Load frame paths
    frame_paths = load_frames_from_dir(frames_dir)
    if not frame_paths:
        # Extract frames from video if no frames found
        print(f"No frames found in {frames_dir}. Extracting frames from video...")
        frame_paths = extract_frames_from_video(video_path, frames_dir)

    # Load first frame
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError("Could not read first frame")

    # Get or load points
    if os.path.exists(points_file):
        print("Loading saved points...")
        points, labels = load_points(points_file)
    else:
        print("Select points on the first frame...")
        points, labels = get_interactive_points(first_frame)
        if points:
            save_points(points, labels, points_file)
        else:
            raise ValueError("No points selected")

    # Format prompts
    prompts = [{
        "point_coords": points,
        "point_labels": labels
    }]

    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print('Using CUDA for inference')
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Setup video writer if saving video
    if save_video:
        if infer_offline:
            output_video_name = os.path.basename(frames_dir)
        else:
            output_video_name = f'webcam_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        output_video_path = os.path.join(results_dir, f'{output_video_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Adjust FPS as needed
        height, width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (img_height, img_width))
        print(f'Output video will be saved to {output_video_path}')

    # Run inference
    if infer_offline:
        infer_offline_video()
    else:
        infer_online_stream()

    print("Processing complete.")
