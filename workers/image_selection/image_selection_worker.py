import os
from workers.base import Base
import datetime
import time
import json
import shutil
import numpy as np
from utils import send_err_email

class ImageSelectionWorker(Base):

    def __init__(self, video_download_dir, video_start, video_end, console_log_level="DEBUG",
                 file_log_level="INFO", log_file=None,
                 max_video_height=960, max_video_width=960,
                  ffmpeg_preset='slow', ffmpeg_preset_webp='default', log_interval=5, log_backup_count=20,
                 clean_folder=True, smart_download='False'):

        self.clean_folder = clean_folder
        self.smart_download = smart_download
        super().__init__(video_download_dir=video_download_dir, video_start=video_start, video_end=video_end, console_log_level=console_log_level, log_file=log_file,
                         file_log_level=file_log_level, max_video_height=max_video_height,
                         max_video_width=max_video_width, ffmpeg_preset=ffmpeg_preset, ffmpeg_preset_webp=ffmpeg_preset_webp,
                         log_interval=log_interval, log_backup_count=log_backup_count)




        # success = self.run(video_id="12345678", file_log=file_log)
        # self.logger.debug(f"success={success}")
    def run(self):
        file_log = {
            "is_info": {
                "type": "is_log"
            },
            "time_start": str(datetime.datetime.utcnow()),
            "process": dict()
        }
        success = self.process_video(video_id="123", file_log=file_log)
        self.logger.debug(f"success={success}")


    def process_video(self, video_id, file_log):
        run_path = None
        upload_mp4_file = None
        tic = time.time()
        success = False

        try:
            # setup run_path
            run_path = self.video_download_path.joinpath(video_id)
            run_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"[process_video][{video_id}] Run path: {run_path}")
            file_log["process"]["run_path"] = str(run_path)

            # download video
            mp4_file = str(run_path.joinpath(f"{video_id}.mp4"))
            self.logger.debug(f"[process_video][{video_id}] mp4_file={mp4_file}")
            file_log["process"]["run_path"] = str(run_path)
            file_log["process"]["mp4_file"] = {"mp4_file": mp4_file, "exist": os.path.exists(mp4_file)}


            # TODO: Check log
            self.logger.info(json.dumps(file_log))

            video_info = self.get_video_info(mp4_file, dimensions=True, fps=True)
            self.logger.debug(f"[process_video][{video_id}] Video info: {video_info}")
            width, height, fps, duration = video_info["width"], video_info["height"], video_info["r_frame_rate"], \
                                           video_info["duration"]
            file_log["process"]["video_info"] = {"width": width, "height": height, "fps": fps}

            self.logger.debug(f"[process_video][{video_id}] width:{width}, height:{height}, fps:{fps}, duration:{duration}")

            frame_start = int(float(video_info['duration']) * self.video_start)
            frame_end = int(float(video_info['duration']) * self.video_end)

            frames, td = self.getHsvInFrames(mp4_file=mp4_file, start_time=frame_start, end_time=frame_end)

            print(f"len(frames)={len(frames)} ,td={td}")

            dis = {}

            def getDistance(hsv1, hsv2):
                # print(f"[getDistance]")
                res = np.sqrt(np.sum(np.square(hsv1-hsv2)))/(1080*1920)
                self.logger.debug(f"[getDistance] res={res}")
                return res

            sec = frame_start
            sec_end = frame_end
            while(sec < sec_end):
                if (frames[sec][2] > 0):
                    dis[sec] = getDistance(frames[sec][1], frames[sec+1][1])
                    self.logger.debug(f"dis[{sec}]={dis[sec]}")
                else:
                    self.logger.debug(f"[WARN] NO FACE! frames[sec][2]={frames[sec][2]}")
                sec += 1


            self.logger.debug(f"len(dis)={len(dis)}")
            self.logger.debug(f"sec={sec}")

            sorted_dis = sorted(dis.items(), key=lambda kv: kv[1], reverse=True)

            self.logger.debug(f"sorted_dis={sorted_dis}")

            # # Process mp4 to jpgs for face detect
            # rank = 1
            # for sec, dis in sorted_dis[:100]:
            #     self.logger.debug(f"sec={sec}, dis={dis}")
            #     (jpg, file_log), td = self.process_mp4_to_jpg(video_id=video_id, mp4_file=mp4_file, run_path=run_path,
            #                                                   start=sec, rank=rank, file_log=file_log)
            #     self.logger.debug(f"[{sec}][{dis}][{rank}] jpg={jpg}")
            #     rank += 1

            # Process mp4 to jpgs for face detect
            rank = 1
            # result = []

            # {k: v for k, v in frames.items() if v[1] < 1}

            # if sec in frames.keys():
            #     if 'faceNem' in frames[sec].keys():
            #         if (frames[sec]['faceNum'] > 0):
            #             result.append()
            for sec, dis in sorted_dis[:100]:
                # self.logger.debug(f"sec={sec}, dis={dis}")
                # if sec in frames.keys():
                #     if 'faceNem' in frames[sec].keys():
                #         if (frames[sec]['faceNum'] > 0):
                #             result.append()

                # (jpg, file_log), td = self.process_mp4_to_jpg(video_id=video_id, mp4_file=mp4_file, run_path=run_path,
                #                                               start=sec, rank=rank, file_log=file_log)
                (jpg, file_log), td = self.process_mp4_to_jpg(video_id=video_id, frames=frames, sec=sec, rank=rank,
                                                              run_path=run_path, file_log=file_log)
                self.logger.debug(f"[{sec}][{dis}][{rank}] jpg={jpg}, td={td}")
                rank += 1

            # file_log["process"]["process_mp4_to_jpg_cmd_result"] = {"td": td, "number_of_jpgs": len(jpgs)}

            # face_detected, td = self.face_detect(image_files=jpgs)
            # file_log["process"]["face_detect"] = {"face_detected": face_detected, "td": td}
            # self.logger.debug(f"[process_video][{video_id}] face_detected:{face_detected}, jpgs:{jpgs}")
            # minify raw mp4 before compression if required
            # self.shot_detect(["video_cache/12345678/frame0136.jpg"])
            # self.shot_detect(jpgs)

            # minify_ops = {}
            # # check dimensions
            # if width <= height and self.max_video_height < height:
            #     # -2 see also: https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
            #     minify_ops["dimensions"] = f"-2:{self.max_video_height}"
            # elif height <= width and self.max_video_width < width:
            #     minify_ops["dimensions"] = f"{self.max_video_width}:-2"
            # check fps
            # if self.max_fps < fps:
            #     minify_ops["fps"] = self.max_fps
            # TODO: Run crf 18 for android video from original cam
            """ffmpeg -i 288265560523836677.mp4 -c:v libx264 -preset veryslow -crf 18 -c:a copy 288265560523836677.18.mp4"""
            # if minify_ops:
            #     try:
            #         (mp4_file, result), td = self.minify_raw_video(mp4_file, minify_ops)
            #         #TODO: What if result != 0 (conversion failed)
            #         if (result != 0):
            #             file_log["process"]["minify"] = {"success": False, "td": td, "result": result}
            #         upload_mp4_file = mp4_file
            #         file_log["process"]["minify"] = {"success": True, "td": td}
            #     except Exception as e:
            #         file_log["process"]["minify"] = {"success": False, "exception": str(e)}
            #         self.logger.error(json.dumps(self.error_msg(video_id, str(e), 'process_recording', "image_selection")))
            #         raise e
            # else:
            #     file_log["process"]["minify"] = {"ignore": True}

            file_log["is_info"]["complete"] = {"success": True}
            success = True

        except Exception as e:
            # don't let the worker stopped by exceptions
            file_log["is_info"]["complete"] = {"success": False, "exception": str(e)}
            self.logger.error(json.dumps(self.error_msg(video_id, str(e), 'process_video', "image_selection")))
        finally:
            #TODO Err msg to queue
            # write file log, success => info,  failed => error
            toc = time.time()
            file_log["time_stop"] = str(datetime.datetime.utcnow())
            file_log["time_elapsed"] = f"{toc - tic:.3f}"
            self.logger.debug(f"[process_video][{video_id}] finally file_log:{file_log}")
            self.logger.debug(f"[process_video][{video_id}] success:{success}, total_time={toc-tic:.3f}")
            self.logger.info(json.dumps(file_log))

            # cleanup local files
            if run_path is not None:
                self.logger.debug(f"[process_video][{video_id}]  Clean_folder={self.clean_folder}, Cleanup {run_path}")
                if (self.clean_folder == 'True'):
                    self.logger.debug(f"[process_video][{video_id}]  Ready to clean up, Cleanup {run_path}")
                    shutil.rmtree(str(run_path))
            return success