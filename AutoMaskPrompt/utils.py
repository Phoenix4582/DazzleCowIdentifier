import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from skimage import measure
import random
from PIL import Image
import json


def preprocess_outputs(output):
    input_scores = [x["score"] for x in output]
    input_labels = [x["label"] for x in output]
    input_boxes = []
    for i in range(len(output)):
        input_boxes.append([*output[i]["box"].values()])
    input_boxes = [input_boxes]
    return input_scores, input_labels, input_boxes


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_boxes_and_labels_on_image(raw_image, boxes, labels, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for i, box in enumerate(boxes):
        show_box(box, plt.gca())
        plt.text(
            x=box[0],
            y=box[1] - 12,
            s=f"{labels[i]}: {scores[i]:,.4f}",
            c="beige",
            path_effects=[pe.withStroke(linewidth=4, foreground="darkgreen")],
        )
    plt.axis("on")
    plt.show()


def show_masks_on_image(raw_image, masks):
    # Create a mask image (assuming binary mask)
    image_with_mask = raw_image.convert("RGBA")
    
    for mask in masks:
        mask = mask.cpu().numpy()
        
        width, height = image_with_mask.size
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]
        
        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)
        
        
        # Overlay the mask on the image
        image_with_mask = Image.alpha_composite(
            image_with_mask,
            mask_image)
    
    # Display the result
    return image_with_mask

def show_multiple_masks_on_image(raw_image, masks, scores):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.array(raw_image))
    for idx in range(len(masks[0])):
        mask = masks[0][idx][0].cpu().detach()
        show_mask(mask, ax, random_color=True)
    ax.axis("off")
    plt.show()


def show_binary_mask(masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    fig, ax = plt.subplots(figsize=(10, 10))
    idx = scores.tolist().index(max(scores))
    mask = masks[idx].cpu().detach()
    ax.imshow(np.array(masks[0, :, :]), cmap="gray")
    score = scores[idx]
    ax.title.set_text(f"Score: {score.item():.3f}")
    ax.axis("off")
    plt.show()


def make_sam_mask(boolean_mask):
    contours = measure.find_contours(boolean_mask, 0.5)
    mask_points = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        mask_points.append(segmentation)
    return mask_points


def make_coco_boxes(detections_boxes):

    """Convert torch tensor Pascal VOC bboxes to COCO format for Comet annotations"""

    list_boxes = detections_boxes
    coco_boxes = [
        [
            list_boxes[0],
            list_boxes[1],
            (list_boxes[2] - list_boxes[0]),
            (list_boxes[3] - list_boxes[1]),
        ]
    ]
    return coco_boxes


def make_bbox_annots(input_scores, input_labels, input_boxes, image_metadata):

    if len(input_boxes[0]) == 0:
        return None

    annotations = [
        {
            "name": "bbox annots",
            "data": [],
            "metadata": image_metadata,
        }
    ]

    for i in range(len(input_boxes[0])):
        annotations[0]["data"].append(
            {
                "label": input_labels[i],
                "score": round((input_scores[i] * 100), 2),
                # bboxes in pascal_voc format, return in coco format for Comet annotations
                "boxes": make_coco_boxes(input_boxes[0][i]),
                "points": None,
            }
        )
    annotations = json.loads(json.dumps(annotations))
    return annotations


def make_mask_annots(input_masks,
                     input_labels,
                     image_metadata
                    ):

    if len(input_masks[0]) == 0:
        return None

    annotations = [
        {
            "name": "mask annots",
            "data": [],
            "meta_data": image_metadata,
        }
    ]

    for i in range(len(input_masks)):
        annotations[0]["data"].append(
            {
                "label": input_labels[i],
                "score": 100.00,
                "points": make_sam_mask(input_masks[i]),
            }
        )
    annotations = json.loads(json.dumps(annotations))
    return annotations

def find_n_highlights(bboxes, n=2, ratio=0.3):
    results = []
    for query in bboxes:
        x_min, y_min, x_max, y_max = query
        box_width = x_max - x_min
        box_height = y_max - y_min
        x_center = int((x_max + x_min) / 2)
        y_center = int((y_max + y_min) / 2)
        points = []
        if n == 1:
            results.append([x_center, y_center])
        else:
            for _ in range(n):
                point_x = x_center + int(random.uniform(-ratio, ratio) * box_width)
                point_y = y_center + int(random.uniform(-ratio, ratio) * box_height)
                points.append([point_x, point_y])
            results.append(points)
    return results

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
