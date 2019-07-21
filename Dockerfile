#FROM python:3.6-alpine
##https://cloud.tencent.com/developer/article/1154600
##RUN apk add yasm && apk add ffmpeg to install ffmpeg
#RUN apk update && apk add --no-cache build-base unzip curl coreutils nasm git cmake zlib-dev jpeg-dev yasm ffmpeg
#
#
#RUN apk add --no-cache x264-dev
#
#COPY requirements.txt /app/
#WORKDIR /app
#RUN pip3.6 install --upgrade pip && pip3.6 install -r requirements.txt
#COPY . /app
##CMD ["python3.6", "start_worker.py"]
#CMD ["tail", "-f"]


# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:3.6-slim
# FROM tensorflow/tensorflow:latest-py3 AS builder
COPY requirements.txt /app/

WORKDIR /app
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 build-essential cmake libgtk-3-dev libboost-all-dev
#sudo apt-get install build-essential cmake
#
#sudo apt-get install libgtk-3-dev
#
#sudo apt-get install libboost-all-dev
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# Using pip:
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip install --upgrade pip && pip install -r requirements.txt
#CMD ["python3.6", "start_worker.py"]
CMD ["tail", "-f"]
