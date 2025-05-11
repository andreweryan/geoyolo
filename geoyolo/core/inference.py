import os
import torch
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from numba import jit
from osgeo import gdal
from ultralytics import YOLO
from typing import List, Tuple, Union

from geoyolo.core.utils import source_images

gdal.UseExceptions()


@jit(nopython=True)
def make_windows(
    src_geotransform: Tuple[float],
    src_width: int,
    src_height: int,
    window_size: int = 512,
    stride: float = 0.2,
) -> List[List[Union[float, int]]]:
    """
    Get sliding window geoinformation for every window in an image (geotransform, image offsets, bounds).

    Args:
        src_geotransform (Tuple[float]): Source image GDAL GeoTransform
        src_width (int): Image width in pixels.
        src_height (int): Image height in pixels.
        window_size (int): Size of sliding window in pixels.
        stride (float): Sliding window overlap, as percentage of window_size, in x and y direction.

    Return:
        geoinfo_list (List[List[Union[float, int]]]): List of lists containing window geotransform and bounds information for all image windows
    """
    geotransform = list(src_geotransform)

    geoinfo_list = list()

    for i in range(0, src_width, int(window_size * (1 - stride))):
        if i + window_size > src_width:
            i = src_width - window_size

        for j in range(0, src_height, int(window_size * (1 - stride))):
            if j + window_size > src_height:
                j = src_height - window_size

            ulx = (
                geotransform[1] * i
                + geotransform[2] * j
                + geotransform[1] * 0.5
                + geotransform[2] * 0.5
                + geotransform[0]
            )
            uly = (
                geotransform[4] * i
                + geotransform[5] * j
                + geotransform[4] * 0.5
                + geotransform[5] * 0.5
                + geotransform[3]
            )

            window_geotransform = [
                ulx,
                geotransform[1],
                geotransform[2],
                uly,
                geotransform[4],
                geotransform[5],
            ]

            lrx = ulx + (window_size * window_geotransform[1])
            lry = uly + (window_size * window_geotransform[5])

            geoinfo = [ulx, lry, lrx, uly, i, j, window_size, window_size]

            geoinfo_list.append(geoinfo)

    return geoinfo_list


def window_transform(
    src_geotransform: Tuple[float, float, float, float, float, float, float, float],
    xoff: int = 0,
    yoff: int = 0,
) -> List[Union[float, int]]:
    """
    Create Affine GeoTransform for sliding window

    Args:
        xoff (int): Column origin coordinate
        yoff (int): Row origin Coordinate
        xsize (int): Width of window, number of columns
        ysize (int): Height of window, number of rows
    Return:
        window_geotransform (List[float]): Geotransform for a window of an image
    """

    x = (
        src_geotransform[1] * xoff
        + src_geotransform[2] * yoff
        + src_geotransform[1] * 0.5
        + src_geotransform[2] * 0.5
        + src_geotransform[0]
    )
    y = (
        src_geotransform[4] * xoff
        + src_geotransform[5] * yoff
        + src_geotransform[4] * 0.5
        + src_geotransform[5] * 0.5
        + src_geotransform[3]
    )

    window_geotransform = [
        x,
        src_geotransform[1],
        src_geotransform[2],
        y,
        src_geotransform[4],
        src_geotransform[5],
    ]

    return window_geotransform


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between all boxes from boxes1 with all boxes from boxes2

    Args:
        boxes1 (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes with format (x1, y1, x2, y2)
        boxes2 (torch.Tensor): Tensor of shape (M, 4) representing bounding boxes with format (x1, y1, x2, y2)

    Returns:
        iou (torch.Tensor): Tensor of shape (N, M) with IoU values for each pair of boxes
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # N
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # M

    # Broadcasting to compute intersection areas for all pairs of boxes
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:4], boxes2[:, 2:4])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + 1e-6)  # [N, M]

    return iou


# need nms to handle oriented bboxes eventually
def non_max_suppression(
    boxes, conf_threshold=0.5, iou_threshold=0.5, max_detections=300
):
    """
    Non-Maximum Suppression NMS on detection boxes.

    Args
        boxes (torch.Tensor): Tensor of shape (N, 6) where each row is [x1, y1, x2, y2, confidence, class]
        iou_threshold (float): IoU threshold for considering a box as a duplicate
        conf_threshold (float): Confidence threshold to filter out low-confidence detections
        max_detections (int): Maximum number of detections to keep

    Returns:
        nms_boxes (torch.Tensor): Filtered tensor containing only the kept boxes
    """
    # Filter out boxes with confidence below threshold
    mask = boxes[:, 4] >= conf_threshold
    boxes = boxes[mask]

    if boxes.shape[0] == 0:
        return torch.zeros((0, 6), device=boxes.device)

    # Sort boxes by confidence (descending)
    _, indices = torch.sort(boxes[:, 4], descending=True)
    boxes = boxes[indices]

    # Apply NMS for each class separately
    unique_classes = boxes[:, 5].unique()
    nms_boxes = []

    for cls in unique_classes:
        # Get boxes of this class
        cls_mask = boxes[:, 5] == cls
        cls_boxes = boxes[cls_mask]

        # Continue while we still have boxes
        while cls_boxes.shape[0] > 0:
            # Take the box with highest confidence
            nms_boxes.append(cls_boxes[0:1])

            # If only one box left, we're done with this class
            if cls_boxes.shape[0] == 1:
                break

            # Calculate IoU of the kept box with all remaining boxes
            ious = box_iou(cls_boxes[0:1, :4], cls_boxes[1:, :4]).squeeze(0)

            # Filter out boxes with IoU >= threshold
            iou_mask = ious < iou_threshold
            cls_boxes = cls_boxes[1:][iou_mask]

    if not nms_boxes:
        return torch.zeros((0, 6), device=boxes.device)

    nms_boxes = torch.cat(nms_boxes, dim=0)

    if nms_boxes.shape[0] > max_detections:
        nms_boxes = nms_boxes[:max_detections]

    return nms_boxes


def detect_image(
    image_path,
    model,
    window_size=1024,
    stride=0.05,
    bands=None,
    confidence=0.5,
    iou=0.5,
    max_detections=10000,
    half=True,
    xyxy=True,
    device=0,
):
    """
    Run inference on single image.

    Args:
        image_path (str): Path to image for inference
        model (): Loaded YOLO model
        window_size (int): Sliding window size
        stride (float): Sliding window overlap in x & y directions
        bands (List[int]): 1-indexed list of 3 band numbers if using MSI imagery
        confidence (float): Confidence threshold
        iou (float): NMS IoU threshold
        max_detections (int): Maximum number of detections allowed per image.
        half (bool): Use FP16 half-precision inference
        xyxy (bool): Return detection bboxes in xyxy format (x1, y1, x2, y2 aka upper left/lower right)

        device (int, str): Device to run inference on
    Returns:

    """

    image = gdal.Open(image_path)
    width = image.RasterXSize
    height = image.RasterYSize
    band_count = image.RasterCount
    src_geotransform = image.GetGeoTransform()

    windows = make_windows(
        src_geotransform,
        width,
        height,
        window_size=window_size,
        stride=stride,
    )

    tensor_list = []

    for window in windows:
        xoff = int(window[4])
        yoff = int(window[5])
        xsize = int(window[6])
        ysize = int(window[7])

        if band_count == 1:  # if single band, convert to 3 band
            window_array = Image.fromarray(
                np.rollaxis(
                    np.array([image.ReadAsArray(xoff, yoff, xsize, ysize)] * 3), 0, 3
                )
            )
        if bands:  # select bands to place into R, G, B channels
            window_array = Image.fromarray(
                np.rollaxis(image.ReadAsArray(xoff, yoff, xsize, ysize)[bands], 0, 3)
            )
            # window_array = Image.fromarray(np.rollaxis(image.ReadAsArray(0, 0, 1024, 1024), 0, 3))

        results = model(
            window_array,
            imgsz=window_size,
            conf=confidence,
            iou=iou,
            verbose=False,
            half=half,
            device=device,
        )

        class_map = results[0].names

        boxes = results[0].boxes.xyxy.clone()
        # if OBB need to get the OBB boxes

        boxes[:, 0] += xoff  # x1
        boxes[:, 1] += yoff  # y1
        boxes[:, 2] += xoff  # x2
        boxes[:, 3] += yoff  # y2

        confs = results[0].boxes.conf
        classes = results[0].boxes.cls

        detects = torch.cat([boxes, confs.unsqueeze(1), classes.unsqueeze(1)], dim=1)
        tensor_list.append(detects)
        detects = None

    # """put second step for global NMS here..."""
    detects = torch.cat(tensor_list, dim=0)

    nms_detects = non_max_suppression(
        detects,
        conf_threshold=confidence,
        iou_threshold=iou,
        max_detections=max_detections,
    )

    detects = None

    if xyxy:
        """upper-left lon/lat (x1, y1)"""
        ul_lon = (
            src_geotransform[0]
            + nms_detects[:, 0] * src_geotransform[1]
            + nms_detects[:, 1] * src_geotransform[2]
        )
        ul_lat = (
            src_geotransform[3]
            + nms_detects[:, 0] * src_geotransform[4]
            + nms_detects[:, 1] * src_geotransform[5]
        )

        """lower-right lon/lat (x1, y2)"""
        lr_lon = (
            src_geotransform[0]
            + nms_detects[:, 2] * src_geotransform[1]
            + nms_detects[:, 3] * src_geotransform[2]
        )
        lr_lat = (
            src_geotransform[3]
            + nms_detects[:, 2] * src_geotransform[4]
            + nms_detects[:, 3] * src_geotransform[5]
        )

        geodetections = torch.stack([ul_lon, ul_lat, lr_lon, lr_lat], dim=1)

    else:  # xyxyxyxy
        """upper-left lon/lat (x1, y1)"""
        ul_lon = (
            src_geotransform[0]
            + nms_detects[:, 0] * src_geotransform[1]
            + nms_detects[:, 1] * src_geotransform[2]
        )
        ul_lat = (
            src_geotransform[3]
            + nms_detects[:, 0] * src_geotransform[4]
            + nms_detects[:, 1] * src_geotransform[5]
        )

        """lower-right lon/lat (x2, y2)"""
        lr_lon = (
            src_geotransform[0]
            + nms_detects[:, 2] * src_geotransform[1]
            + nms_detects[:, 3] * src_geotransform[2]
        )
        lr_lat = (
            src_geotransform[3]
            + nms_detects[:, 2] * src_geotransform[4]
            + nms_detects[:, 3] * src_geotransform[5]
        )

        """upper-right lon/lat (x2, y1)"""
        ur_lon = (
            src_geotransform[0]
            + nms_detects[:, 2] * src_geotransform[1]
            + nms_detects[:, 1] * src_geotransform[2]
        )
        ur_lat = (
            src_geotransform[3]
            + nms_detects[:, 2] * src_geotransform[4]
            + nms_detects[:, 1] * src_geotransform[5]
        )

        """lower-left lon/lat (x1, y2)"""
        ll_lon = (
            src_geotransform[0]
            + nms_detects[:, 0] * src_geotransform[1]
            + nms_detects[:, 3] * src_geotransform[2]
        )
        ll_lat = (
            src_geotransform[3]
            + nms_detects[:, 0] * src_geotransform[4]
            + nms_detects[:, 3] * src_geotransform[5]
        )

        """xyxyxyxy: upper left, upper right, lower right, lower left"""
        geodetections = torch.stack(
            [ul_lon, ul_lat, ur_lon, ur_lat, lr_lon, lr_lat, ll_lon, ll_lat], dim=1
        )

    detects = torch.cat([nms_detects, geodetections], dim=1)

    return detects, class_map


def detect(
    src,
    model_path,
    window_size=1024,
    stride=0.05,
    confidence=0.5,
    max_detections=10000,
    half=True,
    xyxy=True,
    device=0,
):
    """
    Main function for detection inference.
    """

    src_images = source_images(src=src)

    model = YOLO(model_path, task="detect")
    model_name = os.path.basename(model_path).split(".")[0]

    with tqdm(total=len(src_images), unit="image") as progress_bar:
        for image_path in src_images:

            progress_bar.set_description(f"{os.path.basename(src).split('.')[0]}")

            detects = detect_image(
                image_path,
                model,
                window_size=window_size,
                stride=stride,
                confidence=confidence,
                max_detections=max_detections,
                half=half,
                xyxy=xyxy,
                device=device,
            )
            # need to export detections here (postgres/gis, geojson, parquet)...
            # for db, need to make connection to db, ensure table exists, if not create it, add primary key
            # header = ["x1", "y1", "x2", "y2", "confidence", "class"]
            # if xyxy:
            #     header + ["ul_lon", "ul_lat", "lr_lon", "lr_lat"]
            # else:
            #     header + ["ul_lon", "ul_lat", "ur_lon", "ur_lat", "lr_lon", "lr_lat", "ll_lon", "ll_lat"]

            progress_bar.update(1)
