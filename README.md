# geoyolo
A high performance object detection inference engine for arbitrarily large satellite imagery

## Installation

GitHub: Clone the repo & run `sh install.sh`
PyPi: `pip install geoyolo`

Assumes GDAL is already installed globally

## TODO:
- Add detection export (postgres/gis, geojson, parquet, bytesio?)
    - Add database connection support, check if export table exists, create if not, add Primary Key
- Add oriented bounding box support (NMS needs to handle rotation as well as OBB boxes vs axis aligned)
- Add typing support where not already
- Add optional base64 encoded detection chip export

## Usage:

```
usage: geoyolo detect [-h] --src SRC --model_path MODEL_PATH [--window_size WINDOW_SIZE] [--stride STRIDE]
                      [--bands [BANDS ...]] [--confidence CONFIDENCE] [--iou IOU] [--classes CLASSES [CLASSES ...]]
                      [--device DEVICE] [--half] [--export {geojson,parquet,database}] [--export_dir EXPORT_DIR]
                      [--database_creds DATABASE_CREDS] [--table TABLE] [--encode_chip]

options:
  -h, --help            show this help message and exit
  --src SRC             Image source, either single image path or directory of images
  --model_path MODEL_PATH
                        Model file path.
  --window_size WINDOW_SIZE
                        Sliding window size.
  --stride STRIDE       Sliding window overlap in horizontal and vertical direction.
  --bands [BANDS ...]   1-indexed bands to use to use for inference.
  --confidence CONFIDENCE
                        Confidence threshold.
  --iou IOU             IOU threshold.
  --classes CLASSES [CLASSES ...]
                        List of YOLO class indices to detect.
  --device DEVICE       Inference device to use, e.g., 0, cpu, mps.
  --half                Run model in fp16/half precision mode.
  --export {geojson,parquet,database}
                        Export format. Options: geojson, parquet, database
  --export_dir EXPORT_DIR
                        Detection export directory for file export.
  --database_creds DATABASE_CREDS
                        Path to JSON containing database information.
  --table TABLE         Database table name.
  --encode_chip         base64 encode detection chip
```
