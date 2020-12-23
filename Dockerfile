FROM python:3.8.7-slim-buster

# apt install needed packages
RUN apt update && apt install -y curl jq r-base

# install node for svgo
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt install -y nodejs

COPY requirements.txt /
RUN pip install -r /requirements.txt
