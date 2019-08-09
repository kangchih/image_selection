import datetime
import logging
import os
import re
import subprocess
from functools import partial
from pathlib import Path
import posixpath
from .helper_func import download_video_from_url, get_video_info
from utils import timer
from logging.handlers import TimedRotatingFileHandler
import cv2
import face_recognition
import time
import matplotlib.pyplot as plt
import dlib
from PIL import Image

class Base:

    def __init__(self, video_download_dir, video_start=0.1, video_end=0.9,
                 console_log_level="DEBUG", log_file=None, file_log_level="INFO",
                 log_interval=5, log_backup_count=20):
        """Lang Dynamic Bitrate Optimizer Base Class

        See also:
        CRF: https://trac.ffmpeg.org/wiki/Encode/H.264
        VMAF: https://github.com/Netflix/vmaf


        :param str video_download_dir: local path for temporarily storing videos
        :param str log_file: file path of the error log file
        :param float max_fps: maximum value of the output fps
        :param str console_log_level: [ERROR|WARNING|INFO|DEBUG]
        :param str file_log_level: [ERROR|INFO]
        """
        # log folder under /app. Ex: /app/log/logfile
        self.log_path = Path('./log')
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(console_log_level)
        self.logger.debug(f"[Base][init] Console log level: {console_log_level}, File log level: {file_log_level}")

        log_handler = TimedRotatingFileHandler(posixpath.join(self.log_path, log_file),
                                               when="m",
                                               interval=log_interval,
                                               backupCount=log_backup_count)
        log_handler.setLevel(file_log_level)
        formatter = logging.Formatter('%(message)s')
        log_handler.setFormatter(formatter)
        self.logger.addHandler(log_handler)
        self.video_download_path = Path(video_download_dir)
        self.video_download_path.mkdir(parents=True, exist_ok=True)
        self.download_video_from_url = partial(download_video_from_url, logger=self.logger)

        self.get_video_info = partial(get_video_info, logger=self.logger)

        self.video_start = video_start
        self.video_end = video_end



    def error_msg(self, id, error_msg, func_name, service_name='webp'):
        err_log = {
            "error_info": {
                "recording_id": id,
                "func_name": func_name,
                "service_name": service_name,
                "error_msg": error_msg
            }

        }
        return err_log


    # @timer
    # def shot_detect(self, image_files):
    #     is_face_detected = False
    #     # self.logger.debug(f"[shot_detect] image_files:{image_files}")
    #
    #     for image_file in image_files:
    #         self.logger.debug(f"[shot_detect] image_file:{image_file}")
    #         channel = None
    #         height = None
    #         width = None
    #         try:
    #             image = cv2.imread(image_file)
    #             # t = type(image)
    #             # self.logger.debug(f"[shot_detect] type:{t}")
    #             if (channel is not None):
    #                 height, width, channel = image.shape
    #
    #                 self.logger.debug(f"[shot_detect] height:{height}")
    #                 self.logger.debug(f"[shot_detect] width:{width}")
    #                 self.logger.debug(f"[shot_detect] channel:{channel}")
    #
    #             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #             self.logger.debug(f"[shot_detect] hsv:{hsv}")
    #             self.logger.debug(f"[shot_detect] len(hsv):{len(hsv)}")
    #
    #             #op1=np.sqrt(np.sum(np.square(vector1-vector2)))
    #
    #             for h in hsv:
    #                 print(f"h={h}")
    #                 print(f"len(h)={len(h)}")
    #                 # len(h)=1920
    #                 # h=[[  0 255 182]
    #                 #  [  0 255 182]
    #                 #  [  0 255 182]
    #                 #  ...
    #                 #  [ 75 160 107]
    #                 #  [ 75 160 107]
    #                 #  [ 75 160 107]]
    #                 for n in h:
    #                     print(f"n={n}")
    #                     print(f"len(n)={len(n)}")
    #                     # n=[116 186 239]
    #                     # len(h)=1920
    #
    #         except FileNotFoundError as e:
    #             self.logger.debug(f"[face_detect] image_file:{image_file} load image error:{e}")
    #             continue
    #     self.logger.debug(f"is_face_detected:{is_face_detected}")
    #     return is_face_detected


    """
    while(1):
    #获取每一帧
    ret,frame = cap.read()
    #转换到HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #设定蓝色的阀值
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    #根据阀值构建掩模
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    #对原图和掩模进行位运算
    res = cv2.bitwise_and(frame,frame,mask=mask)
    #显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
    #关闭窗口
    cv2.destroyAllWindows()
    
    """


    @timer
    def process_mp4_to_jpg(self, video_id, frames, sec, rank, run_path, file_log):
        """"
        Example path:
        video_cache/12345678/frame15.jpg
        """
        self.logger.debug(f"[process_mp4_to_jpg][{video_id}] sec={sec}")
        image = frames[sec][0]
        # self.logger.debug(f"[process_mp4_to_jpg][{video_id}] image={image}")

        cv2.imwrite(f"{run_path}/frame{rank}-{sec}.jpg", image)

        return f"{run_path}/frame{sec}-{rank}.jpg", file_log



    @timer
    def detect_faces(self, image):
        # create a copy of the image to prevent any changes to the original one.
        image_copy = image.copy()

        face_locations = face_recognition.face_locations(image_copy)
        self.logger.debug(f"[face_detect] number of faces: {len(face_locations)}")

        #A list of tuples of found face locations in css (top, right, bottom, left) order

        for faceRect in face_locations:
            self.logger.debug(f"[face_detect] faceRect:{faceRect}")

            # x1 = faceRect.rect.left()
            # y1 = faceRect.rect.top()
            # x2 = faceRect.rect.right()
            # y2 = faceRect.rect.bottom()
            x = faceRect[3]
            y = faceRect[0]
            w = faceRect[1] - x
            h = faceRect[2] - y
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 15)

        return image_copy, len(face_locations), face_locations

    @timer
    def getHsvInFrames(self, mp4_file, start_time, end_time, animation=False):
        vidcap = cv2.VideoCapture(mp4_file)

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            self.logger.debug(f"hasFrames={hasFrames}")

            if hasFrames:
                resized_img = cv2.resize(image, (320, 180))
                # resized_img = cv2.resize(image, (256, 256))

                hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

                faceNum = None
                faceLoc = None
                if (not animation):
                    (img, faceNum, faceLoc), td = self.detect_faces(resized_img)
                    self.logger.debug(f"[getHsvInFrames] faceNum={faceNum}, faceLoc={faceLoc}, td={td}")
                # for test
                # plt.imshow(img)
                # plt.show()

            # height, width, channel = image.shape
            # self.logger.debug(f"[shot_detect] height:{height}, width:{width}, channel:{channel}")

            return hasFrames, image, hsv, faceNum, faceLoc

            # op1=np.sqrt(np.sum(np.square(vector1-vector2)))

            # for h in hsv:
            #     print(f"h={h}")
            #     print(f"len(h)={len(h)}")
            #     # len(h)=1920
            #     # h=[[  0 255 182]
            #     #  [  0 255 182]
            #     #  [  0 255 182]
            #     #  ...
            #     #  [ 75 160 107]
            #     #  [ 75 160 107]
            #     #  [ 75 160 107]]
            #     for n in h:
            #         print(f"n={n}")
            #         print(f"len(n)={len(n)}")
            #         # n=[116 186 239]
            #         # len(h)=1920
        frames = {}
        sec = start_time
        frameRate = 1
        success = True
        while success and sec <= end_time:
            self.logger.debug(f"sec={sec}")
            success, image, hsv, faceNum, faceLoc = getFrame(sec)
            if (success):
                frames[sec] = (image, hsv, faceNum, faceLoc)
            sec = sec + frameRate
            sec = round(sec, 2)

        self.logger.debug(f"len(frames)={len(frames)}")
        self.logger.debug(f"frames[start_time]={frames[start_time]}")
        self.logger.debug(f"frames[end_time]={frames[end_time]}")
        # Release video
        vidcap.release()

        return frames


    ## Example of using other face detect library
    # @timer
    # def test2(self):
    #     vidcap = cv2.VideoCapture('Wildlife.mp4')
    #
    #     def getFrame(sec):
    #         vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    #         hasFrames, image = vidcap.read()
    #         if hasFrames:
    #             cv2.imwrite("frame " + str(sec) + " sec.jpg", image)  # save frame as JPG file
    #         return hasFrames
    #
    #     sec = 0
    #     frameRate = 0.5
    #     success = getFrame(sec)
    #     while success:
    #         sec = sec + frameRate
    #         sec = round(sec, 2)
    #         success = getFrame(sec)

    # @timer
    # def test(self, mp4_file):
    #     # Start default camera
    #     video = cv2.VideoCapture(mp4_file)
    #
    #     # Find OpenCV version
    #     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    #
    #     # With webcam get(CV_CAP_PROP_FPS) does not work.
    #     # Let's see for ourselves.
    #
    #     if int(major_ver) < 3:
    #         fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #         print(f"Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    #     else:
    #         fps = video.get(cv2.CAP_PROP_FPS)
    #         print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    #
    #     # Number of frames to capture
    #     num_frames = 120;
    #
    #     print(f"Capturing {0} frames".format(num_frames))
    #
    #     # Start time
    #     start = time.time()
    #
    #     # Grab a few frames
    #     for i in xrange(0, num_frames):
    #         ret, frame = video.read()
    #
    #     # End time
    #     end = time.time()
    #
    #     # Time elapsed
    #     seconds = end - start
    #     print
    #     "Time taken : {0} seconds".format(seconds)
    #
    #     # Calculate frames per second
    #     fps = num_frames / seconds;
    #     print
    #     "Estimated frames per second : {0}".format(fps);
    #
    #     # Release video
    #     video.release()
    # @timer
    # def detect_faces(self, cascade, image, scaleFactor=1.3):
    #     # create a copy of the image to prevent any changes to the original one.
    #     image_copy = image.copy()
    #
    #     # convert the test image to gray scale as opencv face detector expects gray images
    #     gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    #
    #     # Applying the haar classifier to detect faces
    #     faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=6, minSize=(50, 50),
    #                                           flags=cv2.CASCADE_SCALE_IMAGE)
    #
    #     for (x, y, w, h) in faces_rect:
    #         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 15)
    #
    #     self.logger.debug(f"Faces found: {len(faces_rect)}")
    #
    #     return image_copy

    # @timer
    # def detect_faces(self, cascade, image, scaleFactor=1.3):
    #     # create a copy of the image to prevent any changes to the original one.
    #     image_copy = image.copy()
    #
    #     # convert the test image to gray scale as opencv face detector expects gray images
    #     # gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    #     dnnFaceDetector = dlib.cnn_face_detection_model_v1("./classifier/mmod_human_face_detector.dat")
    #     faceRects = dnnFaceDetector(image, 0)
    #     self.logger.debug(f"Faces found: {len(faceRects)}")
    #     for faceRect in faceRects:
    #         # x1 = faceRect.rect.left()
    #         # y1 = faceRect.rect.top()
    #         # x2 = faceRect.rect.right()
    #         # y2 = faceRect.rect.bottom()
    #         x = faceRect.rect.left()
    #         y = faceRect.rect.top()
    #         w = faceRect.rect.right() - x
    #         h = faceRect.rect.bottom() - y
    #
    #         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 15)
    #
    #     # Applying the haar classifier to detect faces
    #     # faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=6, minSize=(50, 50),
    #     #                                       flags=cv2.CASCADE_SCALE_IMAGE)
    #     #
    #     # for (x, y, w, h) in faces_rect:
    #     #     cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 15)
    #
    #     return image_copy

    def chunkify(self, img, block_width=4, block_height=4):
        shape = img.shape
        x_len = shape[0] // block_width
        y_len = shape[1] // block_height
        # print(x_len, y_len)

        chunks = []
        x_indices = [i for i in range(0, shape[0] + 1, block_width)]
        y_indices = [i for i in range(0, shape[1] + 1, block_height)]

        shapes = list(zip(x_indices, y_indices))

        for i in range(len(shapes)):
            try:
                start_x = shapes[i][0]
                start_y = shapes[i][1]
                end_x = shapes[i + 1][0]
                end_y = shapes[i + 1][1]
                chunks.append(shapes[start_x:end_x][start_y:end_y])
            except IndexError:
                self.logger.debug('End of Array')

        return chunks


    """
        OpenCV Laplace
        void Laplacian(InputArray src, OutputArray dst, int ddepth, int ksize=1, double scale=1, double delta=0, intborderType=BORDER_DEFAULT)
        
        src：輸入圖。
        dst：輸出圖，和輸入圖有相同的尺寸和通道數。
        ddepth：輸出圖的深度，假設輸入圖為CV_8U, 支援CV_8U、CV_16S、CV_32F、CV_64F，假設輸入圖為 CV_16U, 支援CV_16U、CV_32F、CV_64F。
        ksize：核心，預設為1，輸入值必須為正整數。
    
    """
    #TODO Implement Laplacian

    def lapa(self, image, ksize=5):
        gray_lap = cv2.Laplacian(image, cv2.CV_16S, ksize=ksize)
        dst = cv2.convertScaleAbs(gray_lap)

        cv2.imshow('laplacian', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()