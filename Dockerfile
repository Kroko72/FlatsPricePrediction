FROM python:3.11.6-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN apt-get -y update
RUN python3.11 -m pip install --no-cache-dir -r /code/requirements.txt
COPY . /code/app
