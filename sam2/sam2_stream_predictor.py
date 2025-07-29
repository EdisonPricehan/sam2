"""
Object-oriented SAM2 Video/Stream Inference System

This module provides an object-oriented interface for SAM2 video/stream inference,
supporting both offline video processing and real-time streaming with interactive
point selection and performance monitoring.
"""

import torch
import cv2
import os
import json
import time
from glob import glob
from collections import OrderedDict, deque
from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class PointSelector:
    """
    A class to handle interactive point selection on an image using OpenCV.
    Clicking left selects a point as foreground, right selects it as background.
    """
    
    def __init__(self, window_name: str = "Select Points"):
        self.window_name = window_name
        self.image = None
        self.points = []
        self.labels = []
        self._original_image = None

    def reset(self):
        """Reset points and labels for new selection."""
        self.points = []
        self.labels = []
        if self._original_image is not None:
            self.image = self._original_image.copy()

    def select_points(self, frame: np.ndarray) -> Tuple[List[List[int]], List[int]]:
        """
        Get interactive points from the user on a video frame.

        Args:
            frame: The video frame to select points on, [height, width, 3].

        Returns:
            Tuple containing points and labels lists.
        """
        self._original_image = frame.copy()
        self.image = frame.copy()
        self.reset()

        cv2.imshow(self.window_name, self.image)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("Left click for foreground, Right click for background")
        print("Press 'q' when done selecting points")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.points, self.labels

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for point selection.
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click = foreground
            self.points.append([x, y])
            self.labels.append(1)  # Positive click
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.image)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click = background
            self.points.append([x, y])
            self.labels.append(0)  # Negative click
            cv2.circle(self.image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(self.window_name, self.image)


class PerformanceMonitor:
    """
    Monitor and track performance statistics for video processing.
    """
    
    def __init__(self, max_history: int = 50):
        self.processing_times = deque(maxlen=max_history)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_processing_time(self, processing_time: float):
        """
        Add a processing time measurement.
        """
        self.processing_times.append(processing_time)

    def get_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics.
        """
        stats = {
            'avg_processing_time': 0.0,
            'fps': 0.0,
            'memory_usage_mb': 0.0,
        }

        if self.processing_times:
            stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
            stats['fps'] = 1.0 / stats['avg_processing_time'] if stats['avg_processing_time'] > 0 else 0.0

        # Get memory usage
        if self.device.type == "cuda":
            stats['memory_usage_mb'] = torch.cuda.memory_allocated() / 1024 / 1024

        return stats

    def overlay_stats_on_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Display performance statistics on the video frame.
        """
        stats = self.get_stats()
        font = cv2.FONT_HERSHEY_SIMPLEX
        result_frame = frame.copy()

        cv2.putText(result_frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                    font, 0.7, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Time: {stats['avg_processing_time'] * 1000:.1f}ms", (10, 60),
                    font, 0.7, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Memory: {stats['memory_usage_mb']:.1f}MB", (10, 90),
                    font, 0.7, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Frame: {frame_idx}", (10, 120),
                    font, 0.7, (0, 0, 255), 2)

        return result_frame


class VideoFrameProcessor:
    """
    Handle video frame processing, extraction, and visualization.
    """

    @staticmethod
    def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from a video file.

        Args:
            video_path: Path to the video file
            frame_idx: Index of the frame to extract (0-based)

        Returns:
            The extracted frame as a numpy array, or None if extraction failed.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    @staticmethod
    def extract_frames_from_video(video_path: str, frames_dir: str) -> List[str]:
        """
        Extract frames from a video file and save them to a directory.

        Args:
            video_path: Path to the video file
            frames_dir: Directory to save extracted frames

        Returns:
            List of paths to the extracted frames.
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

    @staticmethod
    def load_frames_from_dir(frames_dir: str) -> List[str]:
        """
        Load frame paths sorted by frame number.

        Args:
            frames_dir: Directory containing extracted video frames

        Returns:
            List of sorted frame paths.
        """
        frame_paths = glob(os.path.join(frames_dir, "*.jpg"))
        frame_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        return frame_paths

    @staticmethod
    def preprocess_frame(frame: np.ndarray, img_h: int, img_w: int, device: torch.device) -> torch.Tensor:
        """
        Convert a video frame to a tensor suitable for model input.

        Args:
            frame: The video frame as a numpy array, [height, width, 3].
            img_h: Target height for resizing.
            img_w: Target width for resizing.
            device: The device to move the tensor to (CPU or GPU).

        Returns:
            A tensor of shape [3, img_h, img_w] normalized to [0, 1].
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (img_h, img_w))
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        frame_tensor = frame_tensor.to(device)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        return frame_tensor

    @staticmethod
    def overlay_masks(frame: np.ndarray, masks_logits: torch.Tensor) -> np.ndarray:
        """
        Overlay binary masks on a video frame.

        Args:
            frame: The video frame as a numpy array, [height, width, 3].
            masks_logits: Tensor of shape [num_masks, height, width] with binary mask logits.

        Returns:
            The frame with masks overlaid, as a numpy array.
        """
        result_frame = frame.copy()

        for mask in masks_logits:
            # Convert logits to binary mask
            mask_bool = (mask.cpu().numpy() > 0).squeeze()

            # Resize mask to frame size
            mask_resized = cv2.resize(
                mask_bool.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # Create colored mask overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[mask_resized > 0] = [0, 255, 0]  # Green mask

            # Overlay mask on frame
            result_frame = cv2.addWeighted(result_frame, 1.0, colored_mask, 0.5, 0)

        return result_frame


class StateManager:
    """
    Manage SAM2 inference state including pruning for memory efficiency.
    """
    
    def __init__(self, device: torch.device):
        self.device = device

    def create_initial_state(
            self,
            img_height: int,
            img_width: int,
            first_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Create initial state for online inference.

        Args:
            img_height: Height of the video frames.
            img_width: Width of the video frames.
            first_tensor: Preprocessed tensor of the first frame, shape [3, img_height, img_width].

        Returns:
            A dictionary representing the initial state for inference.
        """
        return {
            "device": self.device,
            "storage_device": self.device,
            "pruned_frames_num": 0,
            "constants": {},
            "video_height": img_height,
            "video_width": img_width,
            "images": first_tensor.unsqueeze(0),
            "num_frames": 1,
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

    def prune_state(
            self,
            state: Dict[str, Any],
            max_frames: int = 10,
    ) -> Dict[str, Any]:
        """
        Prune state to bound GPU memory usage.

        Args:
            state: The current inference state dictionary.
            max_frames: Maximum number of frames to keep in memory.

        Returns:
            The pruned state dictionary with only the last N frames and associated data.
        """
        num_frames = state["num_frames"]
        current_frame = num_frames - 1
        prune_before = current_frame - max_frames

        if num_frames > max_frames:
            state["pruned_frames_num"] = num_frames - max_frames

        # Keep only the last N raw frames
        state["images"] = state["images"][-max_frames:]

        # Prune cached image features
        for idx in list(state["cached_features"]):
            if idx < prune_before:
                del state["cached_features"][idx]

        # Drop old frame metadata
        for obj_meta in state["frames_tracked_per_obj"].values():
            for fidx in list(obj_meta):
                if fidx < prune_before:
                    obj_meta.pop(fidx)

        # Clear old temp outputs
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


class PromptManager:
    """
    Manage saving and loading of prompt points and labels.
    """
    
    @staticmethod
    def save_points(points: List[List[int]], labels: List[int], save_path: str):
        """
        Save prompt points and labels to JSON.

        Args:
            points: List of points where each point is a list of [x, y] coordinates.
            labels: List of labels corresponding to each point (1 for foreground, 0 for background).
            save_path: Path to save the JSON file.
        """
        data = {'points': points, 'labels': labels}
        with open(save_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_points(save_path: str) -> Tuple[List[List[int]], List[int]]:
        """
        Load prompt points and labels from JSON.

        Args:
            save_path: Path to the JSON file containing points and labels.

        Returns:
            Tuple containing points and labels lists.
        """
        with open(save_path, 'r') as f:
            data = json.load(f)
        return data['points'], data['labels']


class SAM2StreamPredictor:
    """
    Main class for SAM2 video inference with support for both offline and online processing.
    """
    
    def __init__(self, model_cfg: str, checkpoint: str, device: Optional[torch.device] = None):
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        
        # Initialize components
        self.predictor: Optional[SAM2VideoPredictor] = None
        self.point_selector = PointSelector()
        self.performance_monitor = PerformanceMonitor()
        self.frame_processor = VideoFrameProcessor()
        self.state_manager = StateManager(self.device)
        self.prompt_manager = PromptManager()
        
        # Initialize predictor
        self._initialize_predictor()

    def _initialize_predictor(self):
        """
        Initialize the SAM2 predictor.
        """
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
        print("SAM2 predictor initialized.")

    def infer_offline_video(
            self,
            video_path: str,
            points: Optional[List[List[int]]] = None,
            labels: Optional[List[int]] = None,
            frames_dir: Optional[str] = None,
            save_video: bool = False,
            output_video_path: Optional[str] = None,
            force_extract: bool = False,
            show_progress: bool = True,
    ):
        """
        Run inference on a video file with automatic frame extraction and management.

        Args:
            video_path: Path to the video file
            points: List of points for object selection (if None, will use interactive selection)
            labels: List of labels corresponding to points
            frames_dir: Directory to save/load extracted frames (if None, auto-generated from video path)
            save_video: Whether to save output video
            output_video_path: Path to save output video
            force_extract: Whether to force re-extraction of frames even if they exist
            show_progress: Whether to display frames during processing
        """
        print("Starting offline video inference...")
        
        # Validate video path
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return

        # Auto-generate frames directory if not provided
        if frames_dir is None:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            video_dir = os.path.dirname(video_path)
            frames_dir = os.path.join(video_dir, f"{video_basename}_frames")

        # Check if frames already exist
        frame_paths = []
        if os.path.exists(frames_dir) and not force_extract:
            frame_paths = self.frame_processor.load_frames_from_dir(frames_dir)
            if frame_paths:
                print(f"Found {len(frame_paths)} existing frames in {frames_dir}")
            else:
                print(f"Frames directory exists but is empty: {frames_dir}")

        # Extract frames if needed
        if not frame_paths or force_extract:
            print(f"Extracting frames from {video_path} to {frames_dir}")
            frame_paths = self.frame_processor.extract_frames_from_video(video_path, frames_dir)
            if not frame_paths:
                print("Failed to extract frames from video")
                return

        # Get points - first try to load from JSON, then interactive selection if needed
        if points is None or labels is None:
            # First, try to load from existing prompt points file
            prompts_file = os.path.join(frames_dir, "prompt_points.json")
            if os.path.exists(prompts_file):
                try:
                    print(f"Loading existing prompt points from {prompts_file}")
                    points, labels = self.prompt_manager.load_points(prompts_file)
                    print(f"Loaded {len(points)} points from saved file")
                except Exception as e:
                    print(f"Failed to load prompt points from {prompts_file}: {e}")
                    points, labels = None, None

            # If no saved points found or loading failed, use interactive selection
            if points is None or labels is None:
                print("No saved points found, using interactive selection...")
                first_frame = cv2.imread(frame_paths[0])
                if first_frame is None:
                    print("Could not load first frame for point selection")
                    return
                points, labels = self.point_selector.select_points(first_frame)

                if not points:
                    print("No points selected. Exiting.")
                    return

                # Save the newly selected points
                self.prompt_manager.save_points(points, labels, prompts_file)
                print(f"Saved prompt points to {prompts_file}")
            else:
                print("Using loaded prompt points from saved file")

        # Initialize video writer if saving
        video_writer = None
        if save_video and output_video_path:
            first_frame = cv2.imread(frame_paths[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, 
                                         (first_frame.shape[1], first_frame.shape[0]))

        with torch.inference_mode():
            print('Initializing state...')
            state = self.predictor.init_state(video_path=frames_dir)
            print('State initialized.')
            print(f'Images shape: {state["images"].shape}')

            # Add initial points
            frame_idx, object_ids, masks_logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
            print(f"Initial frame masks shape: {masks_logits.shape}")

            # Process video frames
            print("Processing video frames...")
            
            for i, (frame_idx, object_ids, masks_logits) in enumerate(self.predictor.propagate_in_video(state)):
                if i >= len(frame_paths):
                    break

                frame = cv2.imread(frame_paths[i])
                if frame is None:
                    continue

                # Create output frame with mask overlay
                output_frame = self.frame_processor.overlay_masks(frame, masks_logits)

                # Show progress with OpenCV display (optional)
                if show_progress:
                    cv2.imshow("SAM2 Offline Processing", output_frame)
                    cv2.waitKey(1)  # Small delay to update display

                # Save video frame if enabled
                if save_video and video_writer:
                    video_writer.write(output_frame)

                print(f"Processed frame {i + 1}/{len(frame_paths)}")
                torch.cuda.empty_cache()

            # Clean up display window
            if show_progress:
                cv2.destroyAllWindows()

            if video_writer:
                video_writer.release()
                print(f"Output video saved to {output_video_path}")

        return frames_dir  # Return the frames directory for potential reuse

    def infer_online_stream(
            self,
            stream: Union[int, str],
            points: Optional[List[List[int]]] = None,
            labels: Optional[List[int]] = None,
            img_height: int = 512,
            img_width: int = 512,
            keep_n_frames: int = 5,
            output_video_path: Optional[str] = None,
    ):
        """
        Run inference on a video stream with real-time processing.
        
        Args:
            stream: Video stream source (webcam index or video file path)
            points: List of points for object selection (if None, will use interactive selection)
            labels: List of labels corresponding to points
            img_height: Target frame height
            img_width: Target frame width
            keep_n_frames: Number of frames to keep in memory
            save_video: Whether to save output video
            output_video_path: Path to save output video
        """
        print("Starting online stream inference...")
        
        # Configure predictor for online processing
        self.predictor.clear_non_cond_mem_around_input = True
        
        # Open video capture
        cap = cv2.VideoCapture(stream)
        if isinstance(stream, int):
            print(f'Using webcam at {cap.get(cv2.CAP_PROP_FPS)} FPS')
        else:
            print(f'Using video file at {cap.get(cv2.CAP_PROP_FPS)} FPS')

        # Get first frame
        ret, frame = cap.read()
        if not ret:
            print("No stream available.")
            return

        frame = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        # Get points - use interactive selection if not provided
        if points is None or labels is None:
            print("No points provided, using interactive selection...")
            points, labels = self.point_selector.select_points(frame)

            if not points:
                print("No points selected. Exiting.")
                cap.release()
                return

        # Initialize video writer if saving
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (img_width, img_height))

        # Build initial state
        first_tensor = self.frame_processor.preprocess_frame(frame, img_height, img_width, self.device)
        state = self.state_manager.create_initial_state(img_height, img_width, first_tensor)
        
        # Get initial features and mask
        self.predictor._get_image_feature(inference_state=state, frame_idx=0, batch_size=1)
        _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        print('Initial points added.')

        try:
            # Main processing loop
            with torch.inference_mode():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    
                    start_time = time.time()
                    
                    # Process frame
                    tensor = self.frame_processor.preprocess_frame(frame, img_height, img_width, self.device)
                    state["images"] = torch.cat((state["images"], tensor.unsqueeze(0)), dim=0)
                    state["num_frames"] += 1
                    frame_id = state["num_frames"] - 1

                    # Prune state to manage memory
                    state = self.state_manager.prune_state(state, max_frames=keep_n_frames)

                    # Run inference
                    self.predictor.propagate_in_video_preflight(state)
                    obj_output_dict = state["output_dict_per_obj"][0]

                    current_out, mask_logits = self.predictor._run_single_frame_inference(
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

                    obj_output_dict["non_cond_frame_outputs"][frame_id] = current_out
                    _, mask_logits = self.predictor._get_orig_video_res_output(state, mask_logits)

                    # Update performance stats
                    processing_time = time.time() - start_time
                    self.performance_monitor.add_processing_time(processing_time)

                    # Create output frame
                    overlaid_frame = self.frame_processor.overlay_masks(frame, mask_logits)
                    output_frame = self.performance_monitor.overlay_stats_on_frame(overlaid_frame, frame_id)

                    # Display frame
                    cv2.imshow("SAM2 Video Stream", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # Save video if enabled
                    if video_writer:
                        video_writer.write(output_frame)

        except Exception as e:
            print(f'Error during inference: {e}')
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
                print(f"Output video saved to {output_video_path}")

    def infer_frame_by_frame(
            self,
            first_frame: np.ndarray,
            points: Optional[List[List[int]]] = None,
            labels: Optional[List[int]] = None,
            img_height: int = 512,
            img_width: int = 512,
            keep_n_frames: int = 5,
    ):
        """
        Generator for frame-by-frame inference, ideal for reinforcement learning applications.
        Yields masks for each frame including the initial frame with selected points.

        Args:
            first_frame: The first frame to initialize the inference, [height, width, 3].
            points: List of points for object selection (if None, will use interactive selection)
            labels: List of labels corresponding to points
            img_height: Target frame height
            img_width: Target frame width
            keep_n_frames: Number of frames to keep in memory for inference
        """
        print("Initializing frame-by-frame inference...")

        # Configure predictor for online processing
        self.predictor.clear_non_cond_mem_around_input = True

        # Resize first frame
        frame = cv2.resize(first_frame, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        # Get points - use interactive selection if not provided
        if points is None or labels is None:
            print("No points provided, using interactive selection...")
            points, labels = self.point_selector.select_points(frame)

            if not points:
                raise ValueError("No points selected. Cannot proceed with inference.")

        # Build initial state
        first_tensor = self.frame_processor.preprocess_frame(frame, img_height, img_width, self.device)
        state = self.state_manager.create_initial_state(img_height, img_width, first_tensor)

        # Get initial features and mask - only SAM2 inference needs no_grad
        with torch.no_grad():
            self.predictor._get_image_feature(inference_state=state, frame_idx=0, batch_size=1)
            _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
            print('Initial points added.')

            # Convert mask to binary numpy array for first frame
            first_mask = (mask_logits[0].cpu().numpy() > 0).squeeze().astype(np.uint8)

        # Yield the first frame mask (outside of no_grad context)
        next_frame = yield frame, first_mask

        frame_count = 1

        # Process subsequent frames
        while next_frame is not None:
            try:
                # Resize incoming frame
                frame = cv2.resize(next_frame, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                # Only SAM2 processing needs no_grad, not the RL policy
                with torch.no_grad():
                    # Process frame
                    tensor = self.frame_processor.preprocess_frame(frame, img_height, img_width, self.device)
                    state["images"] = torch.cat((state["images"], tensor.unsqueeze(0)), dim=0)
                    state["num_frames"] += 1
                    frame_id = state["num_frames"] - 1

                    # Prune state to manage memory
                    state = self.state_manager.prune_state(state, max_frames=keep_n_frames)

                    # Run inference
                    self.predictor.propagate_in_video_preflight(state)
                    obj_output_dict = state["output_dict_per_obj"][0]

                    current_out, mask_logits = self.predictor._run_single_frame_inference(
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

                    obj_output_dict["non_cond_frame_outputs"][frame_id] = current_out
                    _, mask_logits = self.predictor._get_orig_video_res_output(state, mask_logits)

                    # Convert mask to binary numpy array
                    mask = (mask_logits[0].cpu().numpy() > 0).squeeze().astype(np.uint8)

                # Yield mask and wait for next frame (outside of no_grad context)
                next_frame = yield frame, mask
                frame_count += 1

            except Exception as e:
                print(f'Error processing frame {frame_count}: {e}')
                break

        print(f"Processed {frame_count} frames total.")

    def reset_inference_state(self, inference_state=None):
        """
        Reset the inference state properly using SAM2's built-in reset functionality.
        Useful when starting a new episode in RL or switching to a different video.

        Args:
            inference_state: Optional inference state to reset. If None, will reset
                           the predictor's internal state.
        """
        if hasattr(self, 'predictor') and self.predictor is not None:
            if inference_state is not None:
                # Reset the specific inference state
                self.predictor.reset_state(inference_state)
                print("Inference state reset using SAM2's reset_state method.")
            else:
                print("No specific inference state provided to reset.")

            # Clear GPU memory cache
            torch.cuda.empty_cache()
            print("GPU memory cache cleared.")
        else:
            print("No predictor available to reset.")


def frame_by_frame_demo(sam2_inference: SAM2StreamPredictor):
    cap = cv2.VideoCapture(0)  # Use webcam for demo
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1 to minimize buffering

    ret, first_frame = cap.read()
    if ret:
        try:
            # Initialize frame processor with first frame
            frame_processor = sam2_inference.infer_frame_by_frame(
                first_frame,
            )

            # Get mask for first frame
            first_image, first_mask = next(frame_processor)
            print(f"First frame shape: {first_image.shape}, first mask shape: {first_mask.shape}")

            # Convert mask to 3-channel for visualization
            mask_colored = np.stack([first_mask * 255] * 3, axis=-1).astype(np.uint8)
            colored_frame = cv2.addWeighted(first_image, 1, mask_colored, 0.5, 0)
            cv2.imshow('SAM2 Frame-by-Frame Demo', colored_frame)

            # Wait for key press or timeout
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                print("Exiting early...")
            else:
                # Process 10 more frames as example
                for i in range(10):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Send frame and get mask
                    image, mask = frame_processor.send(frame)
                    print(f"Frame {i + 1} mask shape: {mask.shape}, image shape: {image.shape}")

                    # Convert mask to 3-channel for visualization
                    mask_colored = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
                    colored_frame = cv2.addWeighted(image, 1, mask_colored, 0.5, 0)
                    cv2.imshow('SAM2 Frame-by-Frame Demo', colored_frame)

                    # Check for quit key
                    key = cv2.waitKey(1000) & 0xFF
                    if key == ord('q'):
                        print("Exiting early...")
                        break

        except Exception as e:
            print(f"Error in frame-by-frame processing: {e}")
        finally:
            frame_processor.close()
            cv2.destroyAllWindows()

    cap.release()


def main():
    """
    Example usage of the SAM2VideoInference class.
    """
    # Configuration
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
    video_path = '../notebooks/videos/wabash_upstream_fastforward_60x_512x512.mp4'

    # Create inference object
    sam2_inference = SAM2StreamPredictor(model_cfg, checkpoint)

    # Example 1: Offline processing of a video in batched mode
    # sam2_inference.infer_offline_video(video_path)

    # Example 2: Online stream processing from a video without knowing the whole video frames
    # sam2_inference.infer_online_stream(video_path)

    # Example 3: Online stream processing from a webcam
    sam2_inference.infer_online_stream(stream=0, output_video_path='webcam_output.mp4')

    # Example 4: Frame-by-frame processing for RL applications where immediate mask output is needed
    # frame_by_frame_demo(sam2_inference)


if __name__ == "__main__":
    main()
