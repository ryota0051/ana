FROM python:3.9.7-buster

RUN apt update && apt upgrade -y && \
    apt install -y libopencv-dev

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

WORKDIR /api
