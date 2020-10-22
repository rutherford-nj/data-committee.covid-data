FROM python:3.9.0-slim-buster

RUN apt update && apt install -y python3-venv python3-pip
COPY requirements.txt /
RUN pip install -r /requirements.txt
