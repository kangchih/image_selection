# Copyright 2018 LangLive All Rights Reserved
import os
import re
import requests
import subprocess
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from utils import timer
import cv2


@timer
def download_video_from_url(url, local_file, logger, chunk_size=255, retries=10):
    logger.debug(f"[download_video_from_url] Download url: {url}")
    s = requests.Session()
    retries = Retry(total=retries, backoff_factor=1, status_forcelist=[502, 503, 504])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    r = s.get(url)
    with open(local_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    if not os.path.isfile(local_file):
        logger.error(f"Download MP4 failed: {url}")
        raise FileNotFoundError(local_file)


def get_video_info(mp4_path, logger):
    """
    Get information of a mp4 file

    :param str mp4_path: mp4 local path
    :param logger.Logger logger: python logger
    :param bool dimensions: return width & height
    :param bool fps: return fps
    :rtype dict
    """

    try:
        logger.debug(f"Get video information mp4_path={mp4_path}")

        vidcap = cv2.VideoCapture(mp4_path)  # 0=camera

        if vidcap.isOpened():
            logger.debug(f"vidcap is opened")

            # get vcap property
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            # it gives me 0.0 :/
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            # duration = vidcap.get(cv2.CAP_PROP_POS_MSEC)
            n_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = n_frames / fps

        logger.debug(f"[get_video_info] width={width}, height={height}, fps={fps}, duration={duration}")
    except Exception as e:
        raise e
    finally:
        vidcap.release()
        return width, height, fps, duration

# def get_video_info(mp4_path, logger, dimensions=True, fps=True, time=True):
#     """
#     Get information of a mp4 file
#
#     :param str mp4_path: mp4 local path
#     :param logger.Logger logger: python logger
#     :param bool dimensions: return width & height
#     :param bool fps: return fps
#     :rtype dict
#     """
#     keys = []
#     if dimensions:
#         keys.append("width")
#         keys.append("height")
#     if fps:
#         keys.append("r_frame_rate")
#     if time:
#         keys.append("duration")
#
#     try:
#         logger.debug(f"Get video information: {mp4_path}")
#         cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream={','.join(keys)} -of csv=p=0 {mp4_path}"
#         logger.debug(f"[get_video_info] cmd={cmd}")
#         values = subprocess.check_output(re.split(r"\s+", cmd)).decode("utf-8").strip().split(",")
#         logger.debug(f"[get_video_info] values={values}")
#         info = dict(zip(keys, values))
#         logger.debug(f"[get_video_info] info={info}")
#     except Exception as e:
#          raise e
#
#     # modify values
#     if dimensions:
#         info["width"], info["height"] = int(info["width"]), int(info["height"])
#     if fps:
#         try:
#             info["r_frame_rate"] = eval(info["r_frame_rate"])
#         except ZeroDivisionError:
#             info["r_frame_rate"] = 0
#     return info
