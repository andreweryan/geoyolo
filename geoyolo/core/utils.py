import os
from typing import List, Tuple, Union


def source_images(src: Union[str, List[str]]) -> List[str]:
    """Find image(s) to run inference.

    Args:
        src (Union[str, List[str]]): Path to a single image file, directory of images, or list of image paths.

    Returns:
        src_list (List[str]): List of image file paths.
    """

    image_types = (".tif", ".nitf", ".ntf", ".TIF", ".NITF", ".NTF")
    image_exts = tuple([x.upper() for x in image_types]) + image_types

    if isinstance(src, list):
        src_list = list()

        for i in src:
            if os.path.isfile(i):
                src_list = [i for i in src if i.endswith(image_exts)]

    elif isinstance(src, str) and os.path.isfile(src):
        src_list = [src]

    elif isinstance(src, str) and os.path.isdir(src):
        src_list = [
            os.path.join(src, i) for i in os.listdir(src) if i.endswith(image_exts)
        ]

    return src_list
