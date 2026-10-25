#!/usr/bin/env python3
"""
Quantitative evaluation of LiDAR-Camera fusion distance estimation
against KITTI ground truth, with a monocular-only baseline comparison.
"""
import os
import json
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
DATA_ROOT = '/home/lenovo/ros2_ws/datasets/kitti/training'
IMAGE_DIR = os.path.join(DATA_ROOT, 'image_2')
VELODYNE_DIR = os.path.join(DATA_ROOT, 'velodyne')
LABEL_DIR = os.path.join(DATA_ROOT, 'label_2')
CALIB_DIR = os.path.join(DATA_ROOT, 'calib')
OUTPUT_JSON = '/home/lenovo/ros2_ws/src/kitti_fusion/results/fusion_metrics.json'

NUM_FRAMES = 100  # matches downloaded subset
IOU_THRESHOLD = 0.3

# Approximate real-world object heights (meters) for monocular baseline
CLASS_HEIGHT = {
    'Car': 1.5, 'Van': 2.0, 'Truck': 3.0,
    'Pedestrian': 1.7, 'Cyclist': 1.7, 'Tram': 3.5
}

# Map YOLO(COCO) class names -> KITTI class names
COCO_TO_KITTI = {
    'car': 'Car', 'truck': 'Truck', 'person': 'Pedestrian', 'bicycle': 'Cyclist'
}
# ------------------------------------------


def parse_calib(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = np.array([float(x) for x in val.strip().split()])
    P2 = data['P2'].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = data['R0_rect'].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :4] = data['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr


def parse_labels(path):
    objs = []
    with open(path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if p[0] == 'DontCare':
                continue
            objs.append({
                'class': p[0],
                'bbox': [float(p[4]), float(p[5]), float(p[6]), float(p[7])],
                'location': np.array([float(p[11]), float(p[12]), float(p[13])])  # camera coords
            })
    return objs


def iou_2d(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def project_lidar_to_image(points_3d, Tr, R0, P2):
    pts_h = np.hstack([points_3d[:, :3], np.ones((points_3d.shape[0], 1))])
    cam = (Tr @ pts_h.T).T
    cam = (R0 @ cam.T).T
    mask = cam[:, 2] > 0
    cam = cam[mask]
    if cam.shape[0] == 0:
        return np.array([]), np.array([]), mask
    proj = (P2 @ cam.T).T
    pts_2d = proj[:, :2] / proj[:, 2:3]
    return pts_2d, cam[:, 2], mask


def detect_3d_objects(points_3d, points_2d, depths, mask, detections_2d, class_names):
    valid = points_3d[mask]
    enriched = np.zeros((valid.shape[0], 6))
    enriched[:, :4] = valid
    enriched[:, 4] = -1
    for det in detections_2d:
        x1, y1, x2, y2 = det['bbox']
        inside = (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) & \
                 (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)
        enriched[inside, 4] = det['class_id']
        enriched[inside, 5] = det['confidence']

    labeled = enriched[enriched[:, 4] >= 0]
    results = []
    for cls_id in np.unique(labeled[:, 4]):
        cls_pts = labeled[labeled[:, 4] == cls_id]
        if len(cls_pts) < 10:
            continue
        clustering = DBSCAN(eps=1.5, min_samples=10).fit(cls_pts[:, :3])
        for lbl in set(clustering.labels_):
            if lbl == -1:
                continue
            cluster = cls_pts[clustering.labels_ == lbl]
            if len(cluster) < 10:
                continue
            center = (np.min(cluster[:, :3], axis=0) + np.max(cluster[:, :3], axis=0)) / 2
            results.append({
                'class_name': class_names[int(cls_id)],
                'center_velo': center
            })
    return results


def main():
    model = YOLO('yolov8n.pt')
    class_names = model.names

    image_files = sorted(os.listdir(IMAGE_DIR))[:NUM_FRAMES]

    fusion_pairs = []   # (pred_distance, gt_distance)
    mono_pairs = []     # (baseline_distance, gt_distance)

    for fname in image_files:
        frame_id = fname.split('.')[0]
        img_path = os.path.join(IMAGE_DIR, fname)
        velo_path = os.path.join(VELODYNE_DIR, frame_id + '.bin')
        label_path = os.path.join(LABEL_DIR, frame_id + '.txt')
        calib_path = os.path.join(CALIB_DIR, frame_id + '.txt')

        if not (os.path.exists(velo_path) and os.path.exists(label_path) and os.path.exists(calib_path)):
            continue

        P2, R0, Tr = parse_calib(calib_path)
        gt_objects = parse_labels(label_path)

        import cv2
        image = cv2.imread(img_path)
        results = model(image, verbose=False)

        detections_2d = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                detections_2d.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': cls_id,
                    'confidence': float(box.conf[0])
                })

        points_3d = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        points_2d, depths, mask = project_lidar_to_image(points_3d, Tr, R0, P2)

        detections_3d = []
        if points_2d.shape[0] > 0:
            detections_3d = detect_3d_objects(points_3d, points_2d, depths, mask, detections_2d, class_names)

        # ---- Fusion: match 3D detections to GT by class + nearest range ----
        for det3d in detections_3d:
            kitti_cls = COCO_TO_KITTI.get(det3d['class_name'], None)
            if kitti_cls is None:
                continue
            center_h = np.append(det3d['center_velo'][:3], 1.0)
            cam_pt = R0 @ (Tr @ center_h)
            pred_dist = np.linalg.norm(cam_pt[:3])

            candidates = [o for o in gt_objects if o['class'] == kitti_cls]
            if not candidates:
                continue
            gt_dists = [np.linalg.norm(o['location']) for o in candidates]
            best_idx = int(np.argmin([abs(g - pred_dist) for g in gt_dists]))
            fusion_pairs.append((pred_dist, gt_dists[best_idx]))

        # ---- Monocular baseline: bbox-height distance estimate ----
        for det in detections_2d:
            kitti_cls = COCO_TO_KITTI.get(class_names[det['class_id']], None)
            if kitti_cls is None or kitti_cls not in CLASS_HEIGHT:
                continue
            x1, y1, x2, y2 = det['bbox']
            bbox_h = max(y2 - y1, 1e-3)
            focal_y = P2[1, 1]
            real_h = CLASS_HEIGHT[kitti_cls]
            mono_dist = (focal_y * real_h) / bbox_h

            candidates = [o for o in gt_objects if o['class'] == kitti_cls
                          and iou_2d([x1, y1, x2, y2], o['bbox']) > IOU_THRESHOLD]
            if not candidates:
                continue
            gt_dist = np.linalg.norm(candidates[0]['location'])
            mono_pairs.append((mono_dist, gt_dist))

    def mae_rmse(pairs):
        if not pairs:
            return None, None, 0
        preds = np.array([p[0] for p in pairs])
        gts = np.array([p[1] for p in pairs])
        errors = preds - gts
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        return float(mae), float(rmse), len(pairs)

    fusion_mae, fusion_rmse, fusion_n = mae_rmse(fusion_pairs)
    mono_mae, mono_rmse, mono_n = mae_rmse(mono_pairs)

    metrics = {
        "dataset": "KITTI 3D Object Detection Benchmark (subset)",
        "num_frames_evaluated": NUM_FRAMES,
        "lidar_camera_fusion": {
            "mae_m": fusion_mae,
            "rmse_m": fusion_rmse,
            "num_matched_objects": fusion_n
        },
        "monocular_baseline": {
            "mae_m": mono_mae,
            "rmse_m": mono_rmse,
            "num_matched_objects": mono_n
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    main()
