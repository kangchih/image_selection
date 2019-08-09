import os
from workers.base import Base
import datetime
import time
import json
import shutil
import numpy as np
from utils import send_err_email
from utils import timer

class ImageSelectionWorker(Base):

    def __init__(self, video_download_dir, video_id, video_start, video_end, animation=0, output_images=100,
                 console_log_level="DEBUG",
                 file_log_level="INFO", log_file=None,
                 log_interval=5, log_backup_count=20,
                 clean_folder=True, smart_download='False'):

        self.clean_folder = clean_folder
        self.smart_download = smart_download
        self.animation = bool(animation)
        self.video_id = video_id
        self.output_images = output_images

        super().__init__(video_download_dir=video_download_dir, video_start=video_start, video_end=video_end,
                         console_log_level=console_log_level, log_file=log_file,
                         file_log_level=file_log_level,
                         log_interval=log_interval, log_backup_count=log_backup_count)

    def run(self):
        file_log = {
            "is_info": {
                "type": "is_log"
            },
            "time_start": str(datetime.datetime.utcnow()),
            "process": dict()
        }
        success, td = self.process_video(video_id=self.video_id, file_log=file_log)
        self.logger.debug(f"success={success}, td={td}")

    @timer
    def process_video(self, video_id, file_log):
        run_path = None
        tic = time.time()
        success = False

        try:
            # setup run_path
            run_path = self.video_download_path.joinpath(video_id)
            run_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"[process_video][{video_id}] Run path: {run_path}, animation: {self.animation}")
            file_log["process"]["run_path"] = str(run_path)
            file_log["process"]["animation"] = self.animation

            # download video
            mp4_file = str(run_path.joinpath(f"{video_id}.mp4"))
            self.logger.debug(f"[process_video][{video_id}] mp4_file={mp4_file}")
            file_log["process"]["mp4_file"] = {"mp4_file": mp4_file, "exist": os.path.exists(mp4_file)}


            # TODO: Check log
            self.logger.info(json.dumps(file_log))

            width, height, fps, duration = self.get_video_info(mp4_file)
            file_log["process"]["video_info"] = {"width": width, "height": height, "fps": fps, "duration": duration}

            self.logger.debug(f"[process_video][{video_id}] width:{width}, height:{height}, fps:{fps}, duration:{duration}")

            frame_start = int(duration * self.video_start)
            frame_end = int(duration * self.video_end)

            frames, td = self.getHsvInFrames(mp4_file=mp4_file, start_time=frame_start, end_time=frame_end,
                                             animation=self.animation)

            self.logger.debug(f"len(frames)={len(frames)} ,td={td}")

            dis = {}

            def getDistance(hsv1, hsv2):
                res = np.sqrt(np.sum(np.square(hsv1-hsv2)))/(int(width)*int(height))
                self.logger.debug(f"[getDistance] res={res}")
                return res

            sec = frame_start
            sec_end = frame_end
            while(sec < sec_end):
                #if faceNum > 0 then get distance and not
                if (self.animation or frames[sec][2] > 0):
                    dis[sec] = getDistance(frames[sec][1], frames[sec+1][1])
                    self.logger.debug(f"dis[{sec}]={dis[sec]}")
                else:
                    self.logger.debug(f"[WARN] NO FACE! frames[sec][2]={frames[sec][2]}")
                sec += 1

            self.logger.debug(f"len(dis)={len(dis)}")
            self.logger.debug(f"sec={sec}")

            sorted_dis = sorted(dis.items(), key=lambda kv: kv[1], reverse=True)

            self.logger.debug(f"sorted_dis={sorted_dis}")

            rank = 1

            for sec, dis in sorted_dis[:self.output_images]:
                # self.logger.debug(f"sec={sec}, dis={dis}")
                (jpg, file_log), td = self.process_mp4_to_jpg(video_id=video_id, frames=frames, sec=sec, rank=rank,
                                                              run_path=run_path, file_log=file_log)
                self.logger.debug(f"[{sec}][{dis}][{rank}] jpg={jpg}, td={td}")
                rank += 1

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