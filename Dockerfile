FROM anatielsantos/tf-cpu
MAINTAINER Anatiel Santos <anatielsantos@gmail.com>

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install ipython

RUN apt update ##[edited]
RUN apt install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN apt -y clean

#RUN mkdir /code
#WORKDIR /code
