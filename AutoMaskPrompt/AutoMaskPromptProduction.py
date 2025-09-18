# Libraries
import os
import shutil
import torch
import random
from PIL import Image
from transformers import pipeline
from ultralytics import SAM
import numpy as np
from tqdm import tqdm
import cv2
from datetime import datetime, timedelta
from multiprocessing import Pool
import functools

from utils import show_masks_on_image
from utils import preprocess_outputs, show_boxes_and_labels_on_image
from utils import find_n_highlights, nms_list

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Adapted from utils.py
def save_masks_and_else(raw_image, masks, mask_folder):
    # Create a mask image (assuming binary mask)
    image_with_mask = raw_image.convert("RGBA")
    reset_folder(os.path.join(mask_folder, "masks"))

    for id, mask in enumerate(masks):
        mask = mask.cpu().numpy()

        width, height = image_with_mask.size
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        mask_raw = mask_array
        # white = [255, 255, 255, 150]
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]

        # mask_raw[mask, :] = white
        mask_raw_image = Image.fromarray(mask)
        mask_raw_image.save(os.path.join(mask_folder, "masks", f"mask_{str(id+1).zfill(3)}.png"))

        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)

        # Overlay the mask on the image
        image_with_mask = Image.alpha_composite(
            image_with_mask,
            mask_image)

    # Save the result at path
    image_with_mask.save(os.path.join(mask_folder, "overlay.png"))

# Using aspect ratio to filter out partial bounding boxes
def filter_invalid_bboxes(outputs, image_shape, min_ratio=0.025, max_ratio=0.1):
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

def extract_frames_from_videos(source:str, dest:str, fps:int=30):
    video_fps=25
    video_base = [item for item in os.listdir(source) if item.endswith(".mp4")]
    for video in video_base:
        video_name = video[:-4]
        video = os.path.join(source, video)
        vidcap = cv2.VideoCapture(video)
        success = True
        count = 0
        while success:
            success, image = vidcap.read()
            if count % fps == 0:
                added_total_seconds = count // video_fps
                new_name = update_time(video_name, added_total_seconds)
                cv2.imwrite(os.path.join(dest, f"{new_name}.jpeg"), image)
            count += 1
        print(f"Successfully extracted from {video} to {dest}.")

def update_time(base_name:str, seconds:int):
    base_datetime = datetime.strptime(base_name, "%Y-%m-%dT%H-%M-%S")
    time_delta = timedelta(seconds=seconds)
    new_datetime = base_datetime + time_delta
    return new_datetime.strftime("%Y-%m-%dT%H-%M-%S")

def process_image(img_path, mask_dir, detector):
    raw_image = Image.open(img_path)
    output = detector(
        raw_image,
        candidate_labels = [text_prompt]
    )
    output = filter_invalid_bboxes(output, raw_image.size)
    input_scores, input_labels, input_boxes = preprocess_outputs(output)
    # Create a list of positive labels of same length as the number of predictions generated above
    labels = np.repeat(1, len(output))
    result = model.predict(
        raw_image,
        bboxes=input_boxes[0],
        labels=labels
    )

    masks = result[0].masks.data
    save_masks_and_else(raw_image, masks, os.path.join(mask_dir, f"{item.split('.')[0]}_masks"))


if __name__ == '__main__':
    detector = pipeline(
        model="google/owlv2-large-patch14-finetuned",
        task="zero-shot-object-detection",
        device=torch.device(0),
        threshold=0.3,
    )

    # What you want to identify in the image
    text_prompt = "cow"

    # Local SAM checkpoint needed
    SAM_version = "mobile_sam.pt"
    SAM2_version = "./weights/SAM/sam2_l.pt"
    # model = SAM(SAM_version)
    model = SAM(SAM2_version)

    base_dir = "./images/raw_mini_full"
    mask_dir = "./images/masks_new_mini"

    min_aspect_ratio = 0.025
    max_aspect_ratio = 0.075
    assert min_aspect_ratio < max_aspect_ratio, "Aspect ratio range must be valid."

    # Number of highlight points per bbox
    num_highlight = 1

    print(f"Min bounding box aspect ratio: {min_aspect_ratio}; Max bounding box aspect ratio: {max_aspect_ratio}")
    # img_paths = [os.path.join(base_dir, item) for item in os.listdir(base_dir)]
    #
    # with Pool(processes=2) as pool:
    #     pool.map(functools.partial(process_image, mask_dir=mask_dir, detector=detector), img_paths)
    for item in tqdm(os.listdir(base_dir), ascii=True, desc="Extracting Masks"):
        img_path = os.path.join(base_dir, item)
        raw_image = Image.open(img_path)
        output = detector(
            raw_image,
            candidate_labels = [text_prompt]
        )

        # Filter 1: Aspect Ratio
        output = filter_invalid_bboxes(output, raw_image.size, min_ratio=min_aspect_ratio, max_ratio=max_aspect_ratio)
        if len(output) == 0:
            print(f"No valid bboxes found on {img_path}. Skip.")
            continue

        input_scores, input_labels, input_boxes = preprocess_outputs(output)
        # Filter 2: NMS
        input_boxes, input_scores = nms_list(input_boxes[0], input_scores, iou_threshold=0.5)
        # Create Highlight Points
        highlight_points = find_n_highlights(input_boxes, n=num_highlight)
        # Create a list of positive labels of same length as the number of predictions generated above
        labels = np.repeat(1, len(input_boxes)) if num_highlight == 1 else np.ones((len(input_boxes),num_highlight))
        result = model.predict(
            raw_image,
            bboxes=input_boxes,
            points=highlight_points,
            labels=labels
        )

        masks = result[0].masks.data
        save_masks_and_else(raw_image, masks, os.path.join(mask_dir, f"{item.split('.')[0]}_masks"))
