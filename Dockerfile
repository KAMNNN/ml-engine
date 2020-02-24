FROM ubuntu
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y wget bzip2
RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    tzdata \
    curl


RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

ENV PATH /root/anaconda3/bin:$PATH

RUN conda update conda
RUN conda update anaconda
RUN conda update --all
RUN bash ./setup.sh
RUN conda activate kamn
RUN python nlp_engine.py