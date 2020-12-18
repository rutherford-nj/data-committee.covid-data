FROM python:3.8.6-slim-buster

# apt install needed packages
RUN apt update && apt install -y curl jq r-base

COPY requirements.txt /
RUN pip install -r /requirements.txt
