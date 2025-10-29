import os
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from ultralytics import YOLO
from shapely.geometry import Polygon
from torchvision.ops import batched_nms
from shapely import from_ragged_array, GeometryType

from geoyolo.core.tileservice import TilingService


def nms(boxes, conf_threshold=0.05, iou_threshold=0.3):
    """
    Non-Maximum Suppression NMS on detection boxes.
    """
    mask = boxes[:, 4] >= conf_threshold
    boxes = boxes[mask]

    if boxes.shape[0] == 0:
        return torch.zeros((0, 6), device=boxes.device)

    box_coords = boxes[:, :4]
    scores = boxes[:, 4]
    classes = boxes[:, 5]

    keep_indices = batched_nms(box_coords, scores, classes, iou_threshold)

    return boxes[keep_indices]


def detect_image(
    image_path,
    model_path,
    device=0,
    window_size=1024,
    stride=0.1,
    bands=None,
    confidence=0.3,
    iou=0.5,
    classes=None,
    max_detections=100000,
    half=True,
    export="geojson",
    export_dir=None,
    batch_size=8,
):
    """
    Run inference on single image with batched processing.

    Args:
        batch_size (int): Number of tiles to process in parallel on GPU
    """
    model = YOLO(model_path, task="detect")
    model_name = os.path.basename(model_path).split(".")[0]

    tiler = TilingService(
        image_path, window_size=window_size, stride=stride, max_queue=batch_size * 2
    )
    src_geotransform = tiler.geotransform

    all_boxes = []
    tile_batch = []
    offset_batch = []
    count = 0
    while True:
        tile = tiler.get_tile()
        count += 1
        if tile is None:
            # Process remaining tiles in batch
            if tile_batch:
                all_boxes.extend(
                    _process_batch(
                        model,
                        tile_batch,
                        offset_batch,
                        window_size,
                        confidence,
                        iou,
                        max_detections,
                        classes,
                        half,
                        device,
                    )
                )
            break

        tile_batch.append(tile["array"])
        offset_batch.append((tile["xoff"], tile["yoff"]))

        # Process when batch is full
        if len(tile_batch) >= batch_size:
            all_boxes.extend(
                _process_batch(
                    model,
                    tile_batch,
                    offset_batch,
                    window_size,
                    confidence,
                    iou,
                    max_detections,
                    classes,
                    half,
                    device,
                )
            )
            tile_batch = []
            offset_batch = []

    if len(all_boxes) == 0:
        # No detections found
        return gpd.GeoDataFrame(
            columns=["confidence", "label", "geometry"], crs=tiler.epsg
        )

    # Merge all detections
    merged_detections = torch.cat(all_boxes, dim=0)
    nms_detects = nms(merged_detections, conf_threshold=confidence, iou_threshold=iou)

    # Move to CPU once for all coordinate calculations
    detects_cpu = nms_detects.cpu().numpy()

    # Vectorized coordinate transformation
    x1, y1, x2, y2 = (
        detects_cpu[:, 0],
        detects_cpu[:, 1],
        detects_cpu[:, 2],
        detects_cpu[:, 3],
    )
    conf = detects_cpu[:, 4]
    cls = detects_cpu[:, 5].astype(int)

    gt = src_geotransform
    ul_lon = gt[0] + x1 * gt[1] + y1 * gt[2]
    ul_lat = gt[3] + x1 * gt[4] + y1 * gt[5]
    lr_lon = gt[0] + x2 * gt[1] + y2 * gt[2]
    lr_lat = gt[3] + x2 * gt[4] + y2 * gt[5]

    # Create polygon coordinates (N, 5, 2)
    coords = np.stack(
        [
            np.column_stack([ul_lon, ul_lat]),
            np.column_stack([lr_lon, ul_lat]),
            np.column_stack([lr_lon, lr_lat]),
            np.column_stack([ul_lon, lr_lat]),
            np.column_stack([ul_lon, ul_lat]),
        ],
        axis=1,
    ).astype("float64")

    # Create geometries efficiently
    n_geoms = coords.shape[0]
    flat_coords = coords.reshape(-1, 2)
    geom_offsets = np.arange(0, n_geoms + 1, dtype=np.int32) * 5
    ring_offsets = np.arange(0, n_geoms + 1, dtype=np.int32)

    geoms = from_ragged_array(
        GeometryType.POLYGON, flat_coords, offsets=(geom_offsets, ring_offsets)
    )

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"confidence": conf, "class": cls, "geometry": geoms}, crs=tiler.epsg
    )

    # Add metadata
    class_map = pd.DataFrame.from_dict(model.names, orient="index", columns=["label"])
    class_map.reset_index(inplace=True)

    gdf = gdf.merge(class_map, left_on="class", right_on="index", how="left")
    gdf.drop(columns=["index", "class"], inplace=True)

    metadata_dict = {
        "image_id": tiler.image_id,
        "image_datetime_utc": tiler.image_datetime,
        "model_name": model_name,
    }
    gdf = gdf.assign(**metadata_dict)

    # Export
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        if export == "geojson":
            gdf.to_file(
                os.path.join(export_dir, f"{tiler.image_id}.geojson"), index=False
            )
    print(count)
    return gdf


def _process_batch(
    model,
    tile_batch,
    offset_batch,
    window_size,
    confidence,
    iou,
    max_detections,
    classes,
    half,
    device,
):
    """Process a batch of tiles through the model."""

    batch_boxes = []

    for tile_array, (xoff, yoff) in zip(tile_batch, offset_batch):
        results = model(
            tile_array,
            imgsz=window_size,
            conf=confidence,
            iou=iou,
            max_det=max_detections,
            classes=classes,
            half=half,
            device=device,
            verbose=False,
        )
        result = results[0]  # single tile
        if len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.clone()
        boxes[:, [0, 2]] += xoff
        boxes[:, [1, 3]] += yoff
        confs = result.boxes.conf
        cls = result.boxes.cls
        detections = torch.cat([boxes, confs.unsqueeze(1), cls.unsqueeze(1)], dim=1)
        batch_boxes.append(detections)

    return batch_boxes
