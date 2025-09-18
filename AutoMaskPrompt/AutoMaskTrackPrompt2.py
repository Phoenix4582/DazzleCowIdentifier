import os
import shutil
import random
from PIL import Image
from transformers import pipeline
from ultralytics import SAM
from ultralytics.models.sam import SAM2VideoPredictor

import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from multiprocessing import Pool
import functools
import cv2
import torch

from utils import show_masks_on_image
from utils import preprocess_outputs, show_boxes_and_labels_on_image

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

def save_rgb_masks(raw_image, masks, save_folder):
    # Create a mask image (assuming binary mask)
    # image_with_mask = raw_image.convert("RGBA")
    reset_dirs(save_folder)

    avg_col_per_row = np.average(raw_image, axis=0)
    avg_col = np.average(avg_col_per_row, axis=0)
    avg_col = avg_col[::-1].astype(np.uint8)
    background = np.full_like(raw_image, avg_col)

    for mask_id, mask in enumerate(masks):
        mask = mask.cpu()
        mask = mask.numpy()
        mask = mask.data.astype('bool')
        mask = np.squeeze(mask)

        mask_foreground = mask == True  # Convert to boolean mask
        mask_background = ~mask_foreground   # Inverse mask for background

        # Apply masks to blend the images
        rgb_mask = np.where(mask_foreground[:, :, None], raw_image, background)

        x_min, y_min, x_max, y_max = extract_bounding_boxes(torch.from_numpy(mask))
        rgb_mask = rgb_mask[y_min:y_max, x_min:x_max]

        rgb_mask = Image.fromarray(rgb_mask, 'RGB')
        rgb_mask.save(os.path.join(save_folder, f"mask_{str(mask_id+1).zfill(3)}.jpeg"))

if __name__ == '__main__':
    from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes
    load_next = LoadImagesAndVideos.__next__
    def new_next(self):
        r = load_next(self)
        self.frame += 1
        self.mode = "video"
        return r
    #detector = pipeline(
    #    model="google/owlv2-large-patch14-finetuned",
    #    task="zero-shot-object-detection",
    #    device=torch.device(0),
    #    threshold=0.3,
    #)
    sam_model = SAM("./weights/SAM/sam2_l.pt")
    video_base = "./videos/2024-10-18T12-44-56.mp4"
    video_base_name = video_base.split("/")[-1][:-4]
    dest_folder = f"./results2/{video_base_name}"
    img_source_base = "./images/raw_mini_full/*"

    LoadImagesAndVideos.__next__ = new_next
    loader = LoadImagesAndVideos(img_source_base)
    loader.source_type = SourceTypes(from_img=True)
    loader.frames = loader.ni
    loader.frame = 0
    loader.mode = "video"
    loader.fps = 25
    #cap = cv2.VideoCapture(video_base)

    #_, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cap.release()
    #pil_image = Image.fromarray(frame)
    #new_boxes, highlight_points = get_owl_boxes(pil_image, detector)

    new_boxes = [[157, 918, 486, 1440], [353, 870, 752, 1440], [102, 481, 307, 1004], [290, 332, 498, 872],
          [288, 1, 521, 379], [491, 66, 693, 545], [497, 626, 828, 1340], [678, 421, 966, 1143],
          [694, 25, 911, 440], [887, 341, 1196, 1049], [883, 32, 1318, 458], [1111, 488, 1428, 1236],
          [1064, 1088, 1580, 1436], [1378, 215, 1662, 944], [1573, 512, 1875, 1224], [1600, 1, 1832, 458],
          [1788, 1, 1957, 376], [1837, 446, 2057, 1088], [1959, 148, 2187, 599], [1805, 1090, 2032, 1440],
          [2023, 777, 2216, 1401], [2206, 752, 2384, 1324], [2264, 407, 2414, 868], [2123, 49, 2271, 447]]

    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
    predictor = SAM2VideoPredictor(overrides=overrides)
    results = predictor(source=loader, bboxes=new_boxes, stream=True)
    for id, r in enumerate(results):
        save_rgb_masks(r.orig_img, r.masks, save_folder=os.path.join(dest_folder, str(id+1).zfill(5)))
        # if id > 60 * 25:
            # break
