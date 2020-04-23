FROM ubuntu:bionic

MAINTAINER Alexey Buzmakov <buzmakov@gmail.com>

ENV TZ=Europe/Moscow

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN DEBIAN_FRONTEND=noninteractive  apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion mc vim && \
    apt-get clean && \
    apt-get autoremove -y && apt-get autoclean -y &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /webrbtm

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

COPY environment.yml /webrbtm/

RUN /opt/conda/bin/conda env create -f environment.yml && /opt/conda/bin/conda clean -y --all 

COPY . /webrbtm
RUN echo "MONGODB_URI = 'mongodb://web_database_1:27017'" > /webrbtm/webrecon/conf.py

CMD bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate xweb && \
           cd /webrbtm/webrecon/ && gunicorn  -t 600 -w 8 -b 0.0.0.0:5550 web_recon:app"

