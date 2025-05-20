import os
import torch
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from osgeo import gdal
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import Polygon
from ultralytics import YOLO
from torchvision.ops import batched_nms
from typing import List, Tuple, Union

from geoyolo.core.utils import source_images

gdal.UseExceptions()


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

    gt = np.array(src_geotransform)

    step = int(window_size * (1 - stride))
    xoffs = np.arange(0, src_width - window_size + 1, step)
    yoffs = np.arange(0, src_height - window_size + 1, step)

    if xoffs[-1] + window_size < src_width:
        xoffs = np.append(xoffs, src_width - window_size)
    if yoffs[-1] + window_size < src_height:
        yoffs = np.append(yoffs, src_height - window_size)

    x, y = np.meshgrid(xoffs, yoffs, indexing="xy")

    x_flat = x.ravel()
    y_flat = y.ravel()

    x_c = x_flat + 0.5
    y_c = y_flat + 0.5

    ulx = gt[1] * x_c + gt[2] * y_c + gt[0]
    uly = gt[4] * x_c + gt[5] * y_c + gt[3]

    lrx = ulx + window_size * gt[1]
    lry = uly + window_size * gt[5]

    geoinfo_array = np.stack(
        [
            ulx,
            lry,
            lrx,
            uly,
            x_flat,
            y_flat,
            np.full_like(x_flat, window_size),
            np.full_like(y_flat, window_size),
        ],
        axis=1,
    )

    geoinfo_list = geoinfo_array.tolist()

    return geoinfo_list


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
def nms(boxes, conf_threshold=0.05, iou_threshold=0.3, max_detections=100000):
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

    mask = boxes[:, 4] >= conf_threshold
    boxes = boxes[mask]

    if boxes.shape[0] == 0:
        return torch.zeros((0, 6), device=boxes.device)

    # Extract box coordinates, scores, and class labels
    box_coords = boxes[:, :4]
    scores = boxes[:, 4]
    classes = boxes[:, 5]

    # Apply class-aware NMS
    keep_indices = batched_nms(box_coords, scores, classes, iou_threshold)

    # Keep top detections if needed
    # if max_detections:
    #     keep_indices = keep_indices[:max_detections]

    return boxes[keep_indices]


def detect_image(
    image_path,
    model,
    device=0,
    window_size=1024,
    stride=0.1,
    bands=None,
    confidence=0.3,
    iou=0.5,
    classes=None,
    max_detections=100000,
    half=True,
    xyxy=True,
    export="geojson",
    export_dir=None,
):
    """
    Run inference on single image.

    Args:
        image_path (str): Path to image for inference
        model (): Loaded YOLO model
        device (int, str): Device to run inference on
        window_size (int): Sliding window size
        stride (float): Sliding window overlap in x & y directions
        bands (List[int]): 1-indexed list of 3 band numbers if using MSI imagery
        confidence (float): Confidence threshold
        iou (float): NMS IoU threshold
        classes (List[int]): Filter detects to a set of class ids. Only those detections will be returned
        max_detections (int): Maximum number of detections allowed per image.
        half (bool): Use FP16 half-precision inference
        xyxy (bool): Return detection bboxes in xyxy format (x1, y1, x2, y2 aka upper left/lower right)
        export (str): Type of export, e.g. geojson, parquet, database
        export_dir (Str): Path to directory for flat file export
    Returns:
        detections (torch.Tensor): Detections tensor
    """

    image = gdal.Open(image_path)
    image_id = os.path.basename(image_path).split(".")[0]
    width = image.RasterXSize
    height = image.RasterYSize
    band_count = image.RasterCount
    info = gdal.Info(image, format="json")
    src_geotransform = image.GetGeoTransform()
    gcps = image.GetGCPs()
    metadata = image.GetMetadata()

    try:
        coord_system = info["coordinateSystem"]["wkt"]
    except ValueError as e:
        coord_system = info["gcps"]["coordinateSyste"]["wkt"]

    if coord_system:
        crs = CRS.from_wkt(coord_system)
        epsg = crs.to_epsg()
    else:
        epsg = 4326

    if src_geotransform[0] == 0 and len(gcps) > 0:
        src_geotransform = gdal.GCPsToGeoTransform(gcps)

    if "TIFFTAG_DATETIME" in metadata.keys():
        date_time = metadata["TIFFTAG_DATETIME"]
        image_datetime = datetime.strptime(date_time, "%Y:%m:%d %H:%M:%S")
    else:
        image_datetime = None

    # image metadata, need to pass this through the return to join with detections results downstream
    metadata_dict = {
        "image_id": image_id,
        "image_datetime_utc": image_datetime,
        "width": width,
        "height": height,
        "geotransform": str(src_geotransform),
        "epsg": epsg,
    }

    # need to join this to detection results to get class label names
    class_map = pd.DataFrame.from_dict(model.names, orient="index", columns=["label"])
    class_map.reset_index(inplace=True)

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

        window_array = image.ReadAsArray(xoff, yoff, xsize, ysize)

        if band_count == 1:  # if single band, convert to 3 band
            window_array = np.array([window_array] * 3)

        if bands:  # select bands to place into R, G, B channels
            window_array = window_array[bands]

        window_array = np.rollaxis(window_array, 0, 3)

        results = model(
            window_array,
            imgsz=window_size,
            conf=confidence,
            iou=iou,
            max_det=max_detections,
            classes=classes,
            half=half,
            device=device,
            verbose=False,
        )

        boxes = results[0].boxes.xyxy.clone()
        # if OBB need to get the OBB boxes

        boxes[:, 0] += xoff  # x1
        boxes[:, 1] += yoff  # y1
        boxes[:, 2] += xoff  # x2
        boxes[:, 3] += yoff  # y2

        confs = results[0].boxes.conf  # confidences
        cls = results[0].boxes.cls  # classes

        detections = torch.cat([boxes, confs.unsqueeze(1), cls.unsqueeze(1)], dim=1)
        tensor_list.append(detections)

    merged_detections = torch.cat(tensor_list, dim=0)

    nms_detects = nms(
        merged_detections,
        conf_threshold=confidence,
        iou_threshold=iou,
        max_detections=max_detections,
    )

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

    else:  # xyxyxyxy will need this + rotation for OBB detector models
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

    # if encode_chip place here, use global xyxy to grab pixels from image

    detects = torch.cat([nms_detects, geodetections], dim=1)

    # need to export detections here (postgres/gis, geojson, parquet)...
    # for db, need to make connection to db, ensure table exists, if not create it, add primary key
    data = []

    header = ["x1", "y1", "x2", "y2", "confidence", "class"]

    if xyxy:
        header = header + ["ul_lon", "ul_lat", "lr_lon", "lr_lat", "geom"]
        for box in detects.cpu().numpy():
            x1, y1, x2, y2, conf, cls, ul_lon, ul_lat, lr_lon, lr_lat = box
            ur_lon, ur_lat = lr_lon, ul_lat
            ll_lon, ll_lat = ul_lon, lr_lat

            coords = [
                (ul_lon, ul_lat),
                (ur_lon, ur_lat),
                (lr_lon, lr_lat),
                (ll_lon, ll_lat),
                (ul_lon, ul_lat),
            ]
            geom = Polygon(coords)
            data.append(
                [x1, y1, x2, y2, conf, int(cls), ul_lon, ul_lat, lr_lon, lr_lat, geom]
            )
    else:
        header = header + [
            "ul_lon",
            "ul_lat",
            "ur_lon",
            "ur_lat",
            "lr_lon",
            "lr_lat",
            "ll_lon",
            "ll_lat",
            "geom",
        ]
        for box in detects.cpu().numpy():
            (
                x1,
                y1,
                x2,
                y2,
                conf,
                cls,
                ul_lon,
                ul_lat,
                ur_lon,
                ur_lat,
                lr_lon,
                lr_lat,
                ll_lon,
                ll_lat,
            ) = box
            coords = [
                (ul_lon, ul_lat),
                (ur_lon, ur_lat),
                (lr_lon, lr_lat),
                (ll_lon, ll_lat),
                (ul_lon, ul_lat),
            ]

            geom = Polygon(coords)
            data.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    int(cls),
                    ul_lon,
                    ul_lat,
                    ur_lon,
                    ur_lat,
                    lr_lon,
                    lr_lat,
                    ll_lon,
                    ll_lat,
                    geom,
                ]
            )

    gdf = gpd.GeoDataFrame(data, columns=header, geometry="geom", crs=epsg)

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    if export == "geojson":
        gdf.to_file(os.path.join(export_dir, f"{image_id}.geojson"), index=False)

    gdf = gdf.merge(class_map, left_on="class", right_on="index", how="left")
    gdf.drop(columns="index", inplace=True)

    gdf = gdf.assign(**metadata_dict)

    # rearranging columns so class and label are next to one another
    cols = list(gdf.columns)
    class_idx = cols.index("class")
    cols.remove("label")
    cols.insert(class_idx + 1, "label")
    gdf = gdf[cols]

    print(gdf.head())
    print(gdf.iloc[999])

    return gdf


def detect(
    src,
    model_path,
    device=0,
    window_size=1024,
    stride=0.1,
    bands=None,
    confidence=0.5,
    iou=0.3,
    classes=None,
    max_detections=100000,
    half=True,
    xyxy=True,
    export="geojson",
    export_dir=None,
):
    """
    Main function for detection inference.
    """

    src_images = source_images(src=src)

    model = YOLO(model_path, task="detect")

    model_name = os.path.basename(model_path).split(".")[0]

    with tqdm(total=len(src_images), unit="image") as progress_bar:
        for image_path in src_images:

            progress_bar.set_description(f"{os.path.basename(src).split(".")[0]}")

            detects = detect_image(
                image_path,
                model,
                device=device,
                window_size=window_size,
                stride=stride,
                bands=bands,
                confidence=confidence,
                iou=iou,
                classes=classes,
                max_detections=max_detections,
                half=half,
                xyxy=xyxy,
                export=export,
                export_dir=export_dir,
            )

            progress_bar.update(1)
