FROM tensorflow/tensorflow
MAINTAINER Anatiel Santos <anatielsantos@gmail.com>

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install ipython

RUN apt -y clean

#RUN mkdir /code
#WORKDIR /code
