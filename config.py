import os
import json

def load_config(file_log={}):
    worker_env = os.getenv("ENV", "dev")
    worker_type = os.getenv("TYPE", "is")
    worker_type_env = worker_env + '-' + worker_type
    secret_file = os.getenv("SECRET_FILE", "./config/dev_is")
    file_log["launch_info"]["worker_type_env"] = worker_type_env
    file_log["launch_info"]["secret_file"] = secret_file

    if secret_file:
        with open(secret_file, 'r') as f:
            configs_from_file = json.load(f)
    else:
        file_log["launch_info"]["secret_file"] = "NotFound"
        raise FileNotFoundError(f"No config files. secret_file={secret_file}")

    conf = configs_from_file
    worker_configs = {
        "dev-is": {
            # optional

            "console_log_level": conf.get("console_log_level", "DEBUG"),
            "file_log_level": conf.get("file_log_level", "INFO"),
            "log_file": conf.get("log_file", 'log'),
            "log_interval": int(conf.get("log_interval", 1)),
            "log_backup_count": int(conf.get("log_backup_count", 20)),
            "video_download_dir": conf.get("video_download_dir", "./video_cache"),
            "max_video_height": int(conf.get("max_video_height", 1280)),
            "max_video_width": int(conf.get("max_video_width", 1280)),
            "video_start": float(conf.get("video_start", 0.1)),
            "video_end": float(conf.get("video_end", 0.9)),
            # "crf_stop": int(conf.get("crf_stop", 33)),
            # "crf_step": int(conf.get("crf_step", 2)),
            "ffmpeg_preset": conf.get("ffmpeg_preset", "veryslow"),
            "clean_folder": conf.get("clean_folder", 'True'),
            "smart_download": conf.get("smart_download", 'False')
        },
        "prod-is": {
            # required
            "mysql_api_host_write": conf.get("mysql_api_host_write"),
            "mysql_api_host_read": conf.get("mysql_api_host_read"),
            "mysql_api_user": conf.get("mysql_api_user"),
            "mysql_api_pwd": conf.get("mysql_api_pwd"),
            "mysql_api_db": conf.get("mysql_api_db"),
            "rbmq_hosts": conf.get("rbmq_hosts"),
            "rbmq_user": conf.get("rbmq_user"),
            "rbmq_pwd": conf.get("rbmq_pwd"),
            "rbmq_vhost": conf.get("rbmq_vhost"),
            "aws_access_key_id": conf.get("aws_access_key_id"),
            "aws_secret_access_key": conf.get("aws_secret_access_key"),

            # optional
            "mysql_api_port": int(conf.get("mysql_api_port", 3306)),
            "rbmq_port": int(conf.get("rbmq_port", 5672)),
            # "inbound_routing_key": conf.get("inbound_routing_key", "video.webp.complete"),
            "inbound_routing_key": conf.get("inbound_routing_key", "video.audit.pass"),
            "inbound_queue": conf.get("inbound_queue", "[data]dbo_worker_source"),
            "tag": conf.get("tag", "c1"),
            "vmaf_threshold": float(conf.get("vmaf_threshold", 90.0)),
            "console_log_level": conf.get("console_log_level", "DEBUG"),
            "file_log_level": conf.get("file_log_level", "INFO"),
            "log_file": conf.get("log_file", 'log'),
            "log_interval": int(conf.get("log_interval", 1)),
            "log_backup_count": int(conf.get("log_backup_count", 20)),
            "video_download_dir": conf.get("video_download_dir", "./video_cache"),
            "max_video_height": int(conf.get("max_video_height", 1280)),
            "max_video_width": int(conf.get("max_video_width", 1280)),
            "max_fps": float(conf.get("max_fps", 30.0)),
            "crf_start": int(conf.get("crf_start", 26)),
            "crf_stop": int(conf.get("crf_stop", 33)),
            "crf_step": int(conf.get("crf_step", 2)),
            "ffmpeg_preset": conf.get("ffmpeg_preset", "veryslow"),
            "s3_download_bucket": conf.get("s3_download_bucket", 'langlive-video'),
            "s3_download_key_prefix": conf.get("s3_download_key_prefix", 'prod/recording'),
            "s3_upload_bucket": conf.get("s3_upload_bucket", 'langlive-video'),
            "s3_upload_key_prefix": conf.get("s3_upload_key_prefix", 'prod/recording'),
            "clean_folder": conf.get("clean_folder", 'True'),
            "smart_download": conf.get("smart_download", 'False')
        }
    }
    return worker_type, worker_configs[worker_type_env], file_log