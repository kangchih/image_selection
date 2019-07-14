import logging
import os
import re
import subprocess
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

log_level = os.getenv("IS_LOGLEVEL", "INFO")
log_id = os.getenv("IS_LOGID", __name__)
logging.basicConfig()
logger = logging.getLogger(log_id)
logger.setLevel(log_level)

def get_dimensions(mp4_path):
    """
    Get height and width of a mp4 file

    :param mp4_path: mp4 local path
    :return: (str) (height, width)
    """
    try:
        logger.info(f"get_dimensions: {mp4_path}")
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {mp4_path}"
        dimensions = subprocess.check_output(re.split(r"\s+", cmd)).decode("utf-8").strip()
        if len(dimensions.split(",")) == 2:
            return dimensions.split(",")
        else:
            logger.error(f"wrong dimension parsed {mp4_path}: {dimensions}")
            return ""
    except:
        logger.error(f"Can'r get dimensions of {mp4_path}.")
        return ""


def timer(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        return result, f"{te - ts:.3f}"

    return timed


def send_err_email(id):
    try:
        sender = ''
        passwd = ''
        receivers = ['kangchihkuo@gmail.com']

        emails = [elem.strip().split(',') for elem in receivers]
        msg = MIMEMultipart()
        msg['Subject'] = f"[ImageSelection][{id}] Error"
        msg['From'] = sender
        msg['To'] = ','.join(receivers)
        part = MIMEText("THIS IS AN AUTOMATED MESSAGE - PLEASE DO NOT REPLY DIRECTLY TO THIS EMAIL")
        msg.attach(part)
        smtp = smtplib.SMTP("smtp.gmail.com:587")
        smtp.ehlo()
        smtp.starttls()
        smtp.login(sender, passwd)
        smtp.sendmail(msg['From'], emails, msg.as_string())
    except Exception as e:
        print(f"Send email error: {e}")