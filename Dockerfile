FROM python:3.8.6-slim-buster

RUN apt update && apt install -y curl jq

COPY requirements.txt /
RUN pip install -r /requirements.txt
