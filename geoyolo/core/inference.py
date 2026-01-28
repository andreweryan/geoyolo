import os
from time import perf_counter
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
from ultralytics import YOLO
from torchvision.ops import batched_nms
from datetime import datetime, timezone
from shapely import from_ragged_array, GeometryType
from typing import List, Literal, Optional, Tuple, Union, Any

from geoyolo.core.tileservice import TilingService
from geoyolo.core.utils import source_images
from geoyolo.core.logger import logger
from geoyolo.core.database import connect, setup_db

logger.propagate = False


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
    src,
    model,
    model_name: str,
    device: int = 0,
    batch_size: int = 8,
    window_size: int = 1024,
    stride: float = 0.1,
    bands: Optional[List[int]] = None,
    confidence: float = 0.3,
    iou: float = 0.5,
    classes: Optional[List[int]] = None,
    max_detections: int = 100000,
    half: bool = True,
    export: Union[Literal["database", "geojson", "parquet"], str] = "geojson",
    export_dir: str = os.path.join(Path.home(), "detects"),
    database_connection=None,
    table: Optional[str] = "detects",
):
    """
    Run inference on single image with batched processing.

    Args:
        batch_size (int): Number of tiles to process in parallel on GPU
    """

    start = perf_counter()

    tiler = TilingService(
        src,
        bands=bands,
        window_size=window_size,
        stride=stride,
        max_queue=batch_size * 2,
    )
    src_geotransform = tiler.geotransform

    all_boxes = []
    tile_batch: List[np.ndarray] = []
    offset_batch: List[Tuple[int, int]] = []

    inference_start = perf_counter()

    while True:
        tile = tiler.get_tile()
        if tile is None:
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
        return gpd.GeoDataFrame(
            columns=["confidence", "label", "geometry"], crs=tiler.epsg
        )

    inference_end = perf_counter()
    logger.info(f"Inference speed: {inference_end - inference_start:.2f} seconds")

    merged_detections = torch.cat(all_boxes, dim=0)

    nms_start = perf_counter()
    nms_detects = nms(merged_detections, conf_threshold=confidence, iou_threshold=iou)
    nms_end = perf_counter()
    logger.info(f"NMS speed: {nms_end - nms_start:.2f} seconds")

    detects_cpu = nms_detects.cpu().numpy()

    x1, y1, x2, y2 = (
        detects_cpu[:, 0],
        detects_cpu[:, 1],
        detects_cpu[:, 2],
        detects_cpu[:, 3],
    )
    conf = detects_cpu[:, 4]
    cls = detects_cpu[:, 5].astype(int)

    affine_start = perf_counter()
    gt = src_geotransform
    ul_lon = gt[0] + x1 * gt[1] + y1 * gt[2]
    ul_lat = gt[3] + x1 * gt[4] + y1 * gt[5]
    lr_lon = gt[0] + x2 * gt[1] + y2 * gt[2]
    lr_lat = gt[3] + x2 * gt[4] + y2 * gt[5]

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
    affine_end = perf_counter()
    logger.info(f"Affine Transform speed: {affine_end - affine_start:.2f} seconds")

    geoproc_start = perf_counter()
    n_geoms = coords.shape[0]
    flat_coords = coords.reshape(-1, 2)
    geom_offsets = np.arange(0, n_geoms + 1, dtype=np.int32) * 5
    ring_offsets = np.arange(0, n_geoms + 1, dtype=np.int32)

    geoms = from_ragged_array(
        GeometryType.POLYGON, flat_coords, offsets=(geom_offsets, ring_offsets)
    )

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
        "processed_date_utc": datetime.now(timezone.utc),
        "model_name": model_name,
    }

    gdf = gdf.assign(**metadata_dict)
    bands_value = [x + 1 for x in bands] if bands else [1, 2, 3]
    gdf["bands"] = ["{" + ",".join(map(str, bands_value)) + "}"] * len(gdf)
    geoproc_end = perf_counter()
    logger.info(f"Geo/postprocessing speed: {geoproc_end - geoproc_start:.2f} seconds")

    # Export
    export_start = perf_counter()
    if export == "database":
        gdf.to_postgis(table, database_connection, if_exists="append", index=False)
    elif export == "geojson":
        export_path = os.path.join(export_dir, f"{tiler.image_id}.geojson")
        gdf.to_file(export_path, index=False)
    elif export == "parquet":
        export_path = os.path.join(export_dir, f"{tiler.image_id}.parquet")
        gdf.to_file(export_path, index=False)
    else:
        print(f"No detections for {src}")
    export_end = perf_counter()
    end = perf_counter()
    logger.info(f"Export speed: {export_end - export_start:.2f} seconds")
    logger.info(f"Total time: {end - start:.2f} seconds")

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


def detect(
    src: Union[str, List[str]],
    model_path: str,
    window_size: int = 1024,
    stride: float = 0.20,
    confidence: float = 0.25,
    iou: float = 0.45,
    classes: Optional[List[int]] = None,
    max_detections: int = 10000,
    export: Union[Literal["geojson", "database", "parquet"], str] = "geojson",
    export_dir: str = os.path.join(Path.home(), "detects"),
    database_creds: Optional[str] = None,
    table: Optional[str] = "detects",
    device: int = 0,
    batch_size: int = 8,
    half: bool = False,
    bands: Optional[List[int]] = None,
) -> None:
    """
    Main function for detection inference.

    Args:
        src (Union[str, List[str]]): Directory path of images, path to single image, or list of image paths
        model_path (str): Path to model
        window_size (int): Size of sliding window
        stride (float): Amount of overlap in x, y direction, e.g., 0.2 for 20% overlap
        confidence (float):  Confidence threshold
        iou (float): NMS IoU threshold
        classes (List[int]): Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned.
        max_detections (int): Maximum number of detections allowed per image.
        export (Union[Literal["geojson", "database", "parquet"], str]): Type of export, options are local geojson, local parquet, or database
        export_dir (str): Directory path to export detections to
        database_creds (str): Credentials to database if pushing detects to database
        table (str): Name of table to push detections to
        device (int): Device number to use for inference
        batch_size (int): Number of tiles to process in parallel on GPU
        half (bool): Use FP16 half-precision inference
        bands (List[int]): 1-indexed list of 3 band numbers if using MSI imagery

    Return:
        None
    """

    src_images = source_images(src=src)

    model = YOLO(model_path, task="detect")
    model_name, model_ext = os.path.basename(model_path).split(".")
    if model_ext == "engine":
        model_format = "TensorRT"
    elif model_ext == "onnx":
        model_format = "ONNX"
    elif model_ext == "torchscript":
        model_format = "TorchScript"
    else:
        model_format = "PyTorch"

    if export == "database":
        if not database_creds:
            raise ValueError("Database credentials not supplied!")
        else:
            database_connection = connect(database_creds, driver="sqlalchemy")
            setup_db(database_connection, detects_table=table)
    if export != "database" and export_dir:
        database_connection = None
        os.makedirs(export_dir, exist_ok=True)

    if bands:
        bands = list(map(int, bands))
        bands = [x - 1 for x in bands]  # go from 1 index to 0 index

    with tqdm(total=len(src_images), unit="image") as progress_bar:
        for src in src_images:
            logger.info(f"image: {src}")
            logger.info(f"model: {model_name}")
            logger.info(f"model format: {model_format}")
            logger.info(f"device: {device}")
            logger.info(f"batch size: {batch_size}")
            bands_value = [x + 1 for x in bands] if bands else [1, 2, 3]
            logger.info(f"bands: {bands_value}")
            logger.info(f"window_size: {window_size}")
            logger.info(f"stride: {stride}")
            logger.info(f"confidence threshold: {confidence}")
            logger.info(f"nms threshold: {iou}")
            logger.info(f"export: {export}")

            progress_bar.set_description(f"{os.path.basename(src).split('.')[0]}")
            detect_image(
                src,
                model,
                model_name,
                device=device,
                batch_size=batch_size,
                window_size=window_size,
                stride=stride,
                confidence=confidence,
                iou=iou,
                classes=classes,
                half=half,
                max_detections=max_detections,
                export=export,
                export_dir=export_dir,
                database_connection=database_connection,
                table=table,
                bands=bands,
            )
            progress_bar.update(1)
