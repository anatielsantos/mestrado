#FROM anatielsantos/tf-cpu
#MAINTAINER Anatiel Santos <anatielsantos@gmail.com>

#COPY requirements.txt /requirements.txt
#RUN pip install -r requirements.txt
#RUN pip install ipython

#RUN apt update ##[edited]
#RUN apt install 'ffmpeg'\
#    'libsm6'\ 
#    'libxext6'  -y

#RUN apt -y clean

##RUN mkdir /code
##WORKDIR /code

#----------------------------------------------------
FROM jonnison/tf-1.4-gpu
LABEL authors="Anatiel Santos <anatielsantos@gmail.com>, Jonnison Lima <jonnison.1234@gmail.com>"

RUN apt update ##[edited]
RUN apt install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN apt -y clean

#RUN mkdir /code
#WORKDIR /code
