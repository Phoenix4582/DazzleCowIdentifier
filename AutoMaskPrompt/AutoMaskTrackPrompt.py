import os
import shutil
import random
from PIL import Image
from transformers import pipeline
from ultralytics import SAM
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from multiprocessing import Pool
import functools
import cv2
import torch
from collections import deque
import matplotlib.pyplot as plt

from utils import show_masks_on_image
from utils import preprocess_outputs, show_boxes_and_labels_on_image

# --- Config ---
REDETECT_EVERY = 25
DISAPPEAR_TOLERANCE = 25 * 10
IOU_THRESHOLD = 0.5

# --- Tracked Object Class ---
class TrackedObject:
    def __init__(self, obj_id, box, mask, frame_idx):
        self.id = obj_id
        self.box = box
        self.mask = mask
        self.last_seen = frame_idx
        self.history = deque(maxlen=DISAPPEAR_TOLERANCE)

    def update(self, box, mask, frame_idx):
        self.box = box
        self.mask = mask
        self.last_seen = frame_idx
        self.history.append(True)

    def mark_missed(self):
        self.history.append(False)

    def is_alive(self):
        return sum(self.history) > 0 or len(self.history) < DISAPPEAR_TOLERANCE

# --- Helper Functions ---
def reset_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_owl_boxes(frame, detector, text_prompt="cow"):
    output = detector(
        frame,
        candidate_labels = [text_prompt]
    )
    output = filter_invalid_bboxes(output, frame.size)
    input_scores, input_labels, input_boxes = preprocess_outputs(output)
    input_boxes, input_scores = nms_list(input_boxes[0], input_scores, iou_threshold=0.5)
    highlight_points = find_n_highlights(input_boxes, n=2)
    # # What you want to identify in the image
    # text_prompt = "cow"
    return input_boxes, highlight_points
    # # Placeholder: insert your OWL-ViT detection code here
    # return [[100, 150, 200, 250]]  # dummy box: [x1, y1, x2, y2]

def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

# Using aspect ratio to filter out partial bounding boxes
def filter_invalid_bboxes(outputs, image_shape, min_ratio=0.025, max_ratio=0.075):
    filtered_indices = []
    img_height, img_width = image_shape

    for idx, query in enumerate(outputs):
        x_min, y_min, x_max, y_max = query['box'].values()
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Calculate area ratio relative to the full frame
        area_ratio = (box_width * box_height) / (img_width * img_height)

        # Keep only large enough objects
        if area_ratio >= min_ratio and area_ratio <= max_ratio:
            filtered_indices.append(query)

    return filtered_indices

def compute_iou(box1, box2):
    """Computes IoU between two boxes: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

def nms_list(boxes, scores, iou_threshold=0.5):
    """
    Perform NMS on a list of boxes and scores.

    Args:
        boxes (List[List[float]]): List of [x1, y1, x2, y2]
        scores (List[float]): Confidence scores
        iou_threshold (float): IoU threshold for suppression

    Returns:
        kept_boxes: list of boxes after NMS
        kept_scores: list of corresponding scores
    """
    # Sort by score descending
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices.pop(0)
        keep.append(current)

        indices = [
            i for i in indices
            if compute_iou(boxes[current], boxes[i]) <= iou_threshold
        ]

    kept_boxes = [boxes[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    return kept_boxes, kept_scores

def find_n_highlights(bboxes, n=1, ratio=0.3):
    results = []
    for query in bboxes:
        x_min, y_min, x_max, y_max = query
        box_width = x_max - x_min
        box_height = y_max - y_min
        x_center = int((x_max + x_min) / 2)
        y_center = int((y_max + y_min) / 2)
        results.append([x_center, y_center])
        for _ in range(n-1):
            point_x = x_center + int(random.uniform(-ratio, ratio) * box_width)
            point_y = y_center + int(random.uniform(-ratio, ratio) * box_height)
            results.append([point_x, point_y])
    return results

def extract_bounding_boxes(mask):
    # Find non-zero elements
    non_zero_indices = torch.nonzero(mask)

    if non_zero_indices.numel() == 0:
        return []

    # Get the min and max coordinates
    min_coords = torch.min(non_zero_indices, dim=0)[0]
    max_coords = torch.max(non_zero_indices, dim=0)[0]

    # Bounding box coordinates
    x_min, y_min = min_coords[1].item(), min_coords[0].item()
    x_max, y_max = max_coords[1].item(), max_coords[0].item()

    return x_min, y_min, x_max, y_max

def save_rgb_masks(raw_image, track_objects, save_folder):
    # Create a mask image (assuming binary mask)
    # image_with_mask = raw_image.convert("RGBA")
    reset_dirs(save_folder)

    avg_col_per_row = np.average(raw_image, axis=0)
    avg_col = np.average(avg_col_per_row, axis=0)
    avg_col = avg_col[::-1].astype(np.uint8)
    background = np.full_like(raw_image, avg_col)

    for obj in track_objects:
        mask = obj.mask
        mask_id = obj.id
        mask = mask.astype('bool')
        mask = np.squeeze(mask)

        mask_foreground = mask == True  # Convert to boolean mask
        mask_background = ~mask_foreground   # Inverse mask for background

        # Apply masks to blend the images
        rgb_mask = np.where(mask_foreground[:, :, None], raw_image, background)

        x_min, y_min, x_max, y_max = extract_bounding_boxes(torch.from_numpy(mask))
        rgb_mask = rgb_mask[y_min:y_max, x_min:x_max]

        rgb_mask = Image.fromarray(rgb_mask, 'RGB')
        rgb_mask.save(os.path.join(save_folder, f"mask_{str(mask_id+1).zfill(3)}.jpeg"))

def nms_masks(masks, boxes, iou_thresh=0.7):
    keep = []
    used = set()
    scores = [np.sum(mask) for mask in masks]  # proxy: area of mask

    # Sort indices by score descending
    sorted_idx = np.argsort(scores)[::-1]

    for i in sorted_idx:
        if i in used:
            continue
        keep.append(i)
        for j in sorted_idx:
            if j == i or j in used:
                continue
            iou = mask_iou(masks[i], masks[j])
            if iou > iou_thresh:
                used.add(j)
    return keep

def extract_masks(video, bbox_detector, sam_model, save_path, fps=25, max_time=60):
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    next_id = 0
    tracked_objects = []
    metadata_query = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Wrap as PIL.Image
        pil_image = Image.fromarray(frame)
        # --- Decide to re-detect new targets
        if frame_idx == 0 or frame_idx % REDETECT_EVERY == 0:
            # Get bounding boxes from OWL; Area of Interest, Bounding box sizes, and NMS are all applied
            new_boxes, _ = get_owl_boxes(pil_image, detector)
        else:
            # --- Propagate existing boxes (naive: reuse previous)
            new_boxes = [obj.box for obj in tracked_objects if obj.is_alive()]
        if len(new_boxes) > 0:
            sam_results = sam_model.predict(pil_image, bboxes=new_boxes, retina_masks=True)
            masks = sam_results[0].masks.data.cpu().numpy()
            boxes = sam_results[0].boxes.xyxy.cpu().numpy()
            # Apply NMS to reduce redundant masks
            keep_idx = nms_masks(masks, boxes, iou_thresh=0.7)
            masks = [masks[i] for i in keep_idx]
            # --- Match new masks with existing ones using IoU
            matched_ids = set()
            for i, new_mask in enumerate(masks):
                new_box = new_boxes[i]
                best_iou = 0
                best_obj = None
                for obj in tracked_objects:
                    iou = mask_iou(new_mask, obj.mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_obj = obj

                if best_iou > IOU_THRESHOLD:
                    best_obj.update(new_box, new_mask, frame_idx)
                    matched_ids.add(best_obj.id)
                else:
                    # New object
                    tracked_objects.append(TrackedObject(next_id, new_box, new_mask, frame_idx))
                    next_id += 1

        # --- Mark missed objects
        for obj in tracked_objects:
            if obj.last_seen < frame_idx:
                obj.mark_missed()

        # --- Filter out dead tracks
        tracked_objects = [obj for obj in tracked_objects if obj.is_alive()]

        # --- Draw masks
        display = frame.copy()
        save_rgb_masks(frame, tracked_objects, save_folder=os.path.join(save_path, str(frame_idx).zfill(5)))
        metadata_query.append([int(obj.id) for obj in tracked_objects])
        # for obj in tracked_objects:
        #     color = (int(obj.id * 37) % 255, int(obj.id * 59) % 255, int(obj.id * 23) % 255)
        #     mask = obj.mask > 0.5
        #     display[mask] = display[mask] * 0.5 + np.array(color) * 0.5
        #     x1, y1, x2, y2 = map(int, obj.box)
        #     cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(display, f"ID:{obj.id}", (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # cv2.imshow("SAM2 Tracker", display)
        # reset_dirs(os.path.join(save_path, str(frame_idx).zfill(5)))
        # cv2.imwrite(os.path.join(save_path, str(frame_idx).zfill(5), "overview.jpeg"), display)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_idx += 1
        if frame_idx >= fps * max_time:
            break

    cap.release()
    cv2.destroyAllWindows()

    x = list(range(frame_idx))
    y1 = [len(item) for item in metadata_query]
    plt.figure(figsize=(12, 6))
    plt.xlabel("Timestamp Frame ID")
    plt.ylabel("Number of Cropped Items")
    plt.plot(x, y1, "o--")
    plt.title("OWL+SAM2 Tracker Metadata")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(save_path, "metadata.png"), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # --- Initialize ---
    sam_model = SAM("./weights/SAM/sam2_l.pt")
    video_path = "./videos/2024-10-18T12-44-56.mp4"
    detector = pipeline(
        model="google/owlv2-large-patch14-finetuned",
        task="zero-shot-object-detection",
        device=torch.device(0),
        threshold=0.3,
    )

    video_base_name = video_path.split("/")[-1][:-4]
    save_path = f"./results/{video_base_name}"
    extract_masks(video_path, bbox_detector=detector, sam_model=sam_model, save_path=save_path, fps=25, max_time=60)
