import json
import os
from typing import List

# dataset location

WHOLE_BODY_ANNOTAION_PATH = (
    "/data/user18302289/handpose-keypoints/coco/wholebody_annotations"
)

WHOLE_BODY_TRAIN = "coco_wholebody_train_v1.0.json"
WHOLE_BODY_VAL = "coco_wholebody_val_v1.0.json"

# OUTPUT_PATH = WHOLE_BODY_ANNOTAION_PATH.replace(
#     "wholebody_annotations", "hand_annotations"
# )

OUTPUT_PATH = "./hand_annotations"

HAND_TRAIN = WHOLE_BODY_TRAIN.replace("wholebody", "hand")
HAND_VAL = WHOLE_BODY_VAL.replace("wholebody", "hand")

## dataset format

handkeypoints_name = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

handkeypoints_skeleton = [
    [0, 1],  # wrist to thumb
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],  # wrist to index_finger
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],  # index_finger to middle_finger
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],  # middle_finger to ring_finger
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],  # wrist to pinky
    [13, 17],  # ring_finger to pinky
    [17, 18],
    [18, 19],
    [19, 20],
]

## convert wholebody to hand


def wholebody2hand(whole_body_annotation) -> List[dict]:
    image_id = whole_body_annotation["image_id"]
    bbox = whole_body_annotation["bbox"]
    area = whole_body_annotation["area"]
    category_id = 1  # 1 for hand

    lefthand_valid = whole_body_annotation["lefthand_valid"]
    lefthand_kpts = whole_body_annotation["lefthand_kpts"]
    lefthand_bbox = whole_body_annotation["lefthand_box"]
    lx, ly, lw, lh = lefthand_bbox
    lefthand_area = lw * lh

    righthand_valid = whole_body_annotation["righthand_valid"]
    righthand_kpts = whole_body_annotation["righthand_kpts"]
    righthand_bbox = whole_body_annotation["righthand_box"]
    rx, ry, rw, rh = righthand_bbox
    righthand_area = rw * rh

    assert len(lefthand_kpts) == len(righthand_kpts)
    assert len(lefthand_kpts) == 21 * 3

    result = []

    if lefthand_valid:
        result.append(
            {
                "id": -1,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": 0,
                "bbox": lefthand_bbox,
                "area": lefthand_area,
                "num_keypoints": 21,
                "keypoints": lefthand_kpts,
            }
        )

    if righthand_valid:
        result.append(
            {
                "id": -1,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": 0,
                "bbox": righthand_bbox,
                "area": righthand_area,
                "num_keypoints": 21,
                "keypoints": righthand_kpts,
            }
        )

    return result


def convert_wholebody(whole_body_data) -> dict:
    info = {
        "description": "COCO-WholeBody Hand Keypoints Dataset",
        "url": "https://github.com/RedContritio/handpose-keypoints",
        "version": "1.0",
        "year": 2023,
        "date_created": "2023/03/13",
    }
    licenses = whole_body_data["licenses"]
    categories = [
        {
            "id": 1,
            "name": "hand",
            "keypoints": handkeypoints_name,
            "skeleton": handkeypoints_skeleton,
        }
    ]

    images = whole_body_data["images"]
    existing_image_ids = set()

    annotations = []

    for annotation in whole_body_data["annotations"]:
        hand_kpts_list = wholebody2hand(annotation)
        for hand_kpts in hand_kpts_list:
            existing_image_ids.add(hand_kpts["image_id"])
            annotations.append(hand_kpts)

    images = [image for image in images if image["id"] in existing_image_ids]
    for i, annotation in enumerate(annotations):
        annotation["id"] = i + 1

    return {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # convert train
    with open(os.path.join(WHOLE_BODY_ANNOTAION_PATH, WHOLE_BODY_TRAIN), "r") as f:
        whole_body_train = json.load(f)

    train_output = convert_wholebody(whole_body_train)
    with open(os.path.join(OUTPUT_PATH, HAND_TRAIN), "w") as f:
        json.dump(train_output, f)

    # convert val
    with open(os.path.join(WHOLE_BODY_ANNOTAION_PATH, WHOLE_BODY_VAL), "r") as f:
        whole_body_val = json.load(f)

    val_output = convert_wholebody(whole_body_val)
    with open(os.path.join(OUTPUT_PATH, HAND_VAL), "w") as f:
        json.dump(val_output, f)
