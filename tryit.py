import torch
import cv2
import os
import json
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor


class PointSelector:
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


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_interactive_points(frame):
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


def display_results(frame, masks_logits, alpha=0.5, fig=None, ax=None):
    """
    Display the frame and masks in a matplotlib figure.
    Args:
        frame (numpy.ndarray): The video frame to display, [height, width, 3].
        masks_logits (torch.Tensor): The masks logits from the model, [num_objects, 1, height, width].
        alpha (float): Transparency for the mask overlay.
        fig (matplotlib.figure.Figure): Optional figure to use for display.
        ax (matplotlib.axes.Axes): Optional axes to use for display.
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


def load_frames_from_dir(frames_dir):
    """Load frame paths sorted by frame number"""
    frame_paths = glob(os.path.join(frames_dir, "*.jpg"))
    frame_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    return frame_paths


def save_points(points, labels, save_path):
    """Save prompt points and labels to JSON"""
    data = {
        'points': points,
        'labels': labels
    }
    with open(save_path, 'w') as f:
        json.dump(data, f)


def load_points(save_path):
    """Load prompt points and labels from JSON"""
    with open(save_path, 'r') as f:
        data = json.load(f)
    return data['points'], data['labels']


if __name__ == '__main__':
    # Model setup
    # checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    checkpoint = "./checkpoints/sam2.1_hiera_small.pt"

    # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # video_path = './notebooks/videos/wabash_upstream_640x480_1fps.mp4'
    # video_path = './notebooks/videos/wabash_upstream_true1fps.mp4'
    # video_path = './notebooks/videos/wabash_upstream_fastforward_60x.mp4'
    # video_path = './notebooks/videos/wabash_upstream_fastforward_60x_128x128.mp4'
    video_path = './notebooks/videos/wabash_upstream_fastforward_60x_512x512.mp4'
    # video_path = './notebooks/videos/bedroom.mp4'

    # frames_dir = "./notebooks/videos/wabash_upstream"
    # frames_dir = "./notebooks/videos/wabash_upstream_128x128"
    frames_dir = "./notebooks/videos/wabash_upstream_512x512"
    # frames_dir = "./notebooks/videos/bedroom"

    # points_file = "notebooks/videos/prompt_points_bedroom.json"
    # points_file = "notebooks/videos/prompt_points_wabash_upstream.json"
    # points_file = "notebooks/videos/prompt_points_wabash_upstream_128x128.json"
    points_file = "notebooks/videos/prompt_points_wabash_upstream_512x512.json"

    results_dir = "./notebooks/videos/results"
    os.makedirs(results_dir, exist_ok=True)

    # Load frame paths
    frame_paths = load_frames_from_dir(frames_dir)
    if not frame_paths:
        # Extract frames from video if no frames found
        print(f"No frames found in {frames_dir}. Extracting frames from video...")
        os.makedirs(frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1
        cap.release()
        print(f"Extracted {len(frame_paths)} frames from video.")

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
    print(f"using device: {device}")

    if device.type == "cuda":
        print('Using CUDA for inference')
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Initialize predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    print("Predictor initialized.")

    # Run inference
    save_video = True  # Set to True if you want to save the output video
    if save_video:
        output_video_path = os.path.join(results_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Adjust FPS as needed
        height, width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with torch.inference_mode():
        print('Initializing state...')
        state = predictor.init_state(video_path=video_path)  # Use video path if you want to process the video directly
        # state = predictor.init_state(video_path=frames_dir)  # Use extracted frames directory
        print('State inited.')

        # Get initial frame results
        frame_idx, object_ids, masks_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,  # The frame index where we choose the points and labels
            obj_id=1,  # A unique integer ID for the object
            points=points,  # Pass points directly
            labels=labels,  # Pass labels directly
        )
        print(f"Initial frame masks shape: {masks_logits.shape}, frame_idx: {frame_idx}, obj_ids: {object_ids}")
        # display_results(first_frame, masks)

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

    print("Processing complete.")
