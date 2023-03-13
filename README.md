# handpose-keypoints
handpose detect based on PaddleDetection keypoints

## install

```bash
pip install -r ppdet/requirements.txt paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
cd ppdet && python setup.py install && cd ..
```

## dataset

images:

- ppdet/dataset/coco/train2017
- ppdet/dataset/coco/val2017

annotations:

- ppdet/dataset/coco/annotations/coco_hand_train_v1.0.json -> hand_annotations/coco_hand_train_v1.0.json
- ppdet/dataset/coco/annotations/coco_hand_val_v1.0.json -> hand_annotations/coco_hand_val_v1.0.json

## train

```bash
cd ppdet
CUDA_VISIBLE_DEVICES=6,7 python3 tools/eval.py -c configs/keypoint/tiny_handpose/tinyhandpose_256x192.yml | tee log/train_256x192.log
```