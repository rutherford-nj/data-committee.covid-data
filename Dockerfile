FROM python:3.9.0-slim-buster

RUN apt update && apt install -y python3-venv python3-pip
RUN pip install matplotlib pandas cycler seaborn
