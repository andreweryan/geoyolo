import os
import datetime
from queue import Queue
from threading import Thread
import numpy as np
from osgeo import gdal
from pyproj import CRS

gdal.UseExceptions()


class TilingService:
    def __init__(self, image_path, window_size=512, stride=0.2, max_queue=10):
        self.image_path = image_path
        self.image_id = os.path.basename(image_path).split(".")[0]
        self.image = gdal.Open(image_path, gdal.GA_ReadOnly)
        self.geotransform = self.image.GetGeoTransform()
        self.width = self.image.RasterXSize
        self.height = self.image.RasterYSize
        self.band_count = self.image.RasterCount
        self.info = gdal.Info(self.image, format="json")
        self.gcps = self.image.GetGCPs()
        self.metadata = self.image.GetMetadata()
        if "coordinateSystem" in self.info.keys():
            coord_system = self.info["coordinateSystem"]["wkt"]
            crs = CRS.from_wkt(coord_system)
            self.epsg = crs.to_epsg()
        elif "coordinateSystem" in self.info["gcps"].keys():
            coord_system = self.info["gcps"]["coordinateSystem"]["wkt"]
            crs = CRS.from_wkt(coord_system)
            self.epsg = crs.to_epsg()

        if self.geotransform[0] == 0 and len(self.gcps) > 0:
            self.geotransform = gdal.GCPsToGeoTransform(self.gcps)

        if "TIFFTAG_DATETIME" in self.metadata.keys():
            date_time = self.metadata["TIFFTAG_DATETIME"]
            self.image_datetime = datetime.strptime(date_time, "%Y:%m:%d %H:%M:%S")
        else:
            self.image_datetime = None

        self.window_size = window_size
        self.stride = stride

        self.queue = Queue(maxsize=max_queue)
        self.stop_signal = object()

        self.windows = self.make_windows()

        self.thread = Thread(target=self._producer, daemon=True)
        self.thread.start()

    def make_windows(self):
        gt = np.array(self.geotransform)
        step = int(self.window_size * (1 - self.stride))
        xoffs = np.arange(0, self.width - self.window_size + 1, step)
        yoffs = np.arange(0, self.height - self.window_size + 1, step)
        if xoffs[-1] + self.window_size < self.width:
            xoffs = np.append(xoffs, self.width - self.window_size)
        if yoffs[-1] + self.window_size < self.height:
            yoffs = np.append(yoffs, self.height - self.window_size)
        x_grid, y_grid = np.meshgrid(xoffs, yoffs, indexing="xy")
        windows = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
        return windows

    def _producer(self):
        """Background thread to read tiles and put them in the queue"""
        for xoff, yoff in self.windows:
            arr = self.image.ReadAsArray(
                int(xoff), int(yoff), self.window_size, self.window_size
            )
            if arr is None:
                continue
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            arr = np.moveaxis(arr, 0, -1)  # HWC
            self.queue.put(
                {
                    "array": arr,
                    "xoff": int(xoff),
                    "yoff": int(yoff),
                    "geotransform": (
                        self.geotransform[0] + xoff * self.geotransform[1],
                        self.geotransform[1],
                        self.geotransform[2],
                        self.geotransform[3] + yoff * self.geotransform[5],
                        self.geotransform[4],
                        self.geotransform[5],
                    ),
                }
            )
        # Signal that we are done
        self.queue.put(self.stop_signal)

    def get_tile(self):
        """Consumer retrieves a tile"""
        tile = self.queue.get()
        if tile is self.stop_signal:
            return None
        return tile
