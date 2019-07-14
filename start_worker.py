from config import load_config
from workers import use_worker
import logging
import os
from pathlib import Path
import posixpath
import datetime
import json


if __name__ == '__main__':
    build_version = '1.0.0'
    file_log_level = os.getenv("file_log_level", "INFO")
    # log folder under /app. Ex: /app/log/logfile
    log_path = Path('./log')
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('__name__')
    logger.setLevel(file_log_level)

    logger.debug(f"[start_worker] file log level: {file_log_level}")
    log_file = 'log'
    file_log = {
        "launch_info": dict(),
        "time_start": str(datetime.datetime.utcnow())
    }

    logger_fh = logging.FileHandler(posixpath.join(log_path, log_file))
    formatter_fh = logging.Formatter('%(message)s')
    logger_fh.setFormatter(formatter_fh)
    logger.addHandler(logger_fh)

    worker_type, worker_config, file_log_config = load_config(file_log)
    file_log["launch_info"]["build_version"] = build_version
    file_log["launch_info"]["video_download_dir"] = worker_config.get("video_download_dir", None)

    file_log["launch_info"]["clean_folder"] = worker_config.get("clean_folder", None)
    file_log["launch_info"]["smart_download"] = worker_config.get("smart_download", None)
    file_log["launch_info"]["log_file"] = worker_config.get("log_file", None)

    logger.info(json.dumps(file_log_config))
    worker = use_worker(worker_type)(**worker_config)
    worker.run()
