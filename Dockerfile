FROM python:3.7-alpine
#https://cloud.tencent.com/developer/article/1154600
#RUN apk add yasm && apk add ffmpeg to install ffmpeg
RUN apk update && apk add --no-cache build-base unzip curl coreutils nasm git cmake zlib-dev jpeg-dev yasm ffmpeg


RUN apk add --no-cache x264-dev

COPY requirements.txt /app/
WORKDIR /app
RUN pip3.7 install -r requirements.txt
COPY . /app
#CMD ["python3.7", "start_worker.py"]
CMD ["tail", "-f"]
