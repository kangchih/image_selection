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
                 max_video_height=960, max_video_width=960,
                 ffmpeg_preset="slow", ffmpeg_preset_webp="default", log_interval=5, log_backup_count=20):
        """Lang Dynamic Bitrate Optimizer Base Class

        See also:
        CRF: https://trac.ffmpeg.org/wiki/Encode/H.264
        VMAF: https://github.com/Netflix/vmaf


        :param str video_download_dir: local path for temporarily storing videos
        :param str log_file: file path of the error log file
        :param int max_video_height: maximum value of the output height
        :param int max_video_width: maximum value of the output width
        # :param float max_fps: maximum value of the output fps
        :param str console_log_level: [ERROR|WARNING|INFO|DEBUG]
        :param str file_log_level: [ERROR|INFO]
        :param int crf_start: starting crf of image_selection testing
        :param int crf_stop: stopping crf of image_selection testing
        :param int crf_step: crf step of image_selection testing
        :param str ffmpeg_preset: [veryslow|slower|slow|default] ffmpeg preset for image_selection testing
        :param str ffmpeg_preset_webp: [photo|picture|drawing|icon|default] ffmpeg preset for webp testing
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
        self.max_video_height = max_video_height
        self.max_video_width = max_video_width
        # self.max_fps = max_fps
        self.download_video_from_url = partial(download_video_from_url, logger=self.logger)

        # self.minify_raw_video = partial(minify_raw_video, logger=self.logger)
        self.get_video_info = partial(get_video_info, logger=self.logger)

        self.video_start = video_start
        self.video_end = video_end

        # CRFparameters
        # if crf_start > crf_stop:
        #     raise ValueError(f"[Base][init] crf_start should be <= crf_stop ({crf_start}, {crf_stop})")
        # self.crf_start = crf_start
        # self.crf_stop = crf_stop
        # if crf_step < 1:
        #     raise ValueError(f"[Base][init] crf_step should be >= 1 ({crf_step})")
        # self.crf_step = crf_step

        # Don't use faster preset
        valid_presets = ["veryslow", "slower", "slow"]
        if ffmpeg_preset not in valid_presets:
            raise ValueError(f"[Base][init] ffmpeg_preset should be one of {valid_presets} ({ffmpeg_preset})")
        self.ffmpeg_preset = ffmpeg_preset
        self.ffmpeg_preset_webp = ffmpeg_preset_webp
        # self.haar_cascade_face = cv2.CascadeClassifier('./classifier/haarcascade_frontalface_default.xml')
        self.haar_cascade_face = cv2.CascadeClassifier('./classifier/haarcascade_frontalface_alt.xml')



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

    @timer
    def face_detect(self, image_files):
        is_face_detected = False
        self.logger.debug(f"[face_detect] image_files:{image_files}")

        for image_file in image_files:
            self.logger.debug(f"[face_detect] image_file:{image_file}")
            try:
                image = face_recognition.load_image_file(image_file)
                face_locations = face_recognition.face_locations(image)
                self.logger.debug(f"[face_detect] face_locations:{face_locations}")
                if (len(face_locations) > 0):
                    self.logger.debug(f"[face_detect] face detected in image_file:{image_file}")
                    is_face_detected = True
                    break
            except FileNotFoundError as e:
                self.logger.debug(f"[face_detect] image_file:{image_file} load image error:{e}")
                continue
        self.logger.debug(f"is_face_detected:{is_face_detected}")
        return is_face_detected

    @timer
    def shot_detect(self, image_files):
        is_face_detected = False
        # self.logger.debug(f"[shot_detect] image_files:{image_files}")

        for image_file in image_files:
            self.logger.debug(f"[shot_detect] image_file:{image_file}")
            channel = None
            height = None
            width = None
            try:
                image = cv2.imread(image_file)
                # t = type(image)
                # self.logger.debug(f"[shot_detect] type:{t}")
                if (channel is not None):
                    height, width, channel = image.shape

                    self.logger.debug(f"[shot_detect] height:{height}")
                    self.logger.debug(f"[shot_detect] width:{width}")
                    self.logger.debug(f"[shot_detect] channel:{channel}")

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                self.logger.debug(f"[shot_detect] hsv:{hsv}")
                self.logger.debug(f"[shot_detect] len(hsv):{len(hsv)}")

                #op1=np.sqrt(np.sum(np.square(vector1-vector2)))

                for h in hsv:
                    print(f"h={h}")
                    print(f"len(h)={len(h)}")
                    # len(h)=1920
                    # h=[[  0 255 182]
                    #  [  0 255 182]
                    #  [  0 255 182]
                    #  ...
                    #  [ 75 160 107]
                    #  [ 75 160 107]
                    #  [ 75 160 107]]
                    for n in h:
                        print(f"n={n}")
                        print(f"len(n)={len(n)}")
                        # n=[116 186 239]
                        # len(h)=1920

                # if (len(face_locations) > 0):
                #     self.logger.debug(f"[face_detect] face detected in image_file:{image_file}")
                #     is_face_detected = True
                #     break
            except FileNotFoundError as e:
                self.logger.debug(f"[face_detect] image_file:{image_file} load image error:{e}")
                continue
        self.logger.debug(f"is_face_detected:{is_face_detected}")
        return is_face_detected


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

    # @timer
    # def process_mp4_to_jpg(self, video_id, mp4_file, run_path, start, rank, file_log):
    #     """"
    #     Example path:
    #     video_cache/12345678/frame15.jpg
    #     """
    #     self.logger.debug(f"[process_mp4_to_jpg][{video_id}] mp4_file={mp4_file}, run_path={run_path}")
    #     # output_file = mp4_file.replace(f'{recording_id}.mp4', output)
    #     cmd = f'ffmpeg -y -i {mp4_file} -loglevel panic -ss {start} -vf fps=1 {run_path}/frame{start}-{rank}.jpg'
    #     self.logger.debug(f"[process_mp4_to_jpg] Process {mp4_file} with command= {cmd}")
    #     res = subprocess.call(cmd, shell=True)
    #     # res = 0
    #     file_log["process"]["mp4_to_jpg_result"] = {"cmd": cmd, "result": res}
    #     self.logger.debug(f"[process_mp4_to_jpg] res = {res}")
    #     # for file in os.listdir(run_path):
    #     #     if file.startswith("frame"):
    #     #         self.logger.debug(f"[process_mp4_to_jpg] {os.path.join(run_path, file)}")
    #     #         result.append(os.path.join(run_path, file))
    #     # return result, file_log
    #     return f"{run_path}/frame{start}-{rank}.jpg", file_log

    @timer
    def process_mp4_to_jpg(self, video_id, frames, sec, rank, run_path, file_log):
        """"
        Example path:
        video_cache/12345678/frame15.jpg
        """
        self.logger.debug(f"[process_mp4_to_jpg][{video_id}] sec={sec}")
        self.logger.debug(f"[process_mp4_to_jpg][{video_id}] frames[sec]={frames[sec]}")
        image = frames[sec][0]
        self.logger.debug(f"[process_mp4_to_jpg][{video_id}] image={image}")

        self.logger.debug(f"[process_mp4_to_jpg][{video_id}] 222222")

        # image_tmp = image.resize((320, 180), Image.ANTIALIAS)
        # self.logger.debug(f"[process_mp4_to_jpg][{video_id}] 333333333")
        cv2.imwrite(f"{run_path}/frame{sec}-{rank}.jpg", image)
        # image.save(f"{run_path}/frame{sec}-{rank}.jpg")
        self.logger.debug(f"[process_mp4_to_jpg][{video_id}] 44444444")

        # output_file = mp4_file.replace(f'{recording_id}.mp4', output)
        # cmd = f'ffmpeg -y -i {mp4_file} -loglevel panic -ss {start} -vf fps=1 {run_path}/frame{start}-{rank}.jpg'
        # self.logger.debug(f"[process_mp4_to_jpg] Process {mp4_file} with command= {cmd}")
        # res = subprocess.call(cmd, shell=True)
        # res = 0
        # file_log["process"]["mp4_to_jpg_result"] = {"cmd": cmd, "result": res}
        # self.logger.debug(f"[process_mp4_to_jpg] res = {res}")
        # for file in os.listdir(run_path):
        #     if file.startswith("frame"):
        #         self.logger.debug(f"[process_mp4_to_jpg] {os.path.join(run_path, file)}")
        #         result.append(os.path.join(run_path, file))
        # return result, file_log
        return f"{run_path}/frame{sec}-{rank}.jpg", file_log

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

    @timer
    def detect_faces(self, cascade, image, scaleFactor=1.3):
        # create a copy of the image to prevent any changes to the original one.
        image_copy = image.copy()

        # convert the test image to gray scale as opencv face detector expects gray images
        # gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

        # image = face_recognition.load_image_file(image)

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

        # for (x, y, w, h) in faces_rect:
        #     cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 15)

        return image_copy, len(face_locations), face_locations




    @timer
    def getHsvInFrames(self, mp4_file, start_time, end_time):
        vidcap = cv2.VideoCapture(mp4_file)

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            print(f"hasFrames={hasFrames}")
            # print(f"image={image}")


            if hasFrames:
                # cv2.imwrite("frame " + str(sec) + " sec.jpg", image)  # save frame as JPG file
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # faces_rects = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
                # for (x, y, w, h) in faces_rects:
                #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #
                # # Let us print the no. of faces found
                # print(f"Faces found: {len(faces_rects)}")
                (img, faceNum, faceLoc), td = self.detect_faces(self.haar_cascade_face, image)
                self.logger.debug(f"[getHsvInFrames] td={td}")
                # for test
                # plt.imshow(img)
                # plt.show()


            # height, width, channel = image.shape
            # self.logger.debug(f"[shot_detect] height:{height}")
            # self.logger.debug(f"[shot_detect] width:{width}")
            # self.logger.debug(f"[shot_detect] channel:{channel}")

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
            print(f"sec={sec}")
            success, image, hsv, faceNum, faceLoc = getFrame(sec)
            if (success):
                frames[sec] = (image, hsv, faceNum, faceLoc)
            sec = sec + frameRate
            sec = round(sec, 2)

        print(f"len(frames)={len(frames)}")
        print(f"frames[start_time]={frames[start_time]}")
        print(f"frames[end_time]={frames[end_time]}")
        # Release video
        vidcap.release()

        return frames

    @timer
    def test2(self):
        vidcap = cv2.VideoCapture('Wildlife.mp4')

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                cv2.imwrite("frame " + str(sec) + " sec.jpg", image)  # save frame as JPG file
            return hasFrames

        sec = 0
        frameRate = 0.5
        success = getFrame(sec)
        while success:
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)