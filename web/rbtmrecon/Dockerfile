FROM nvidia/cuda:10.0-runtime

MAINTAINER Alexey Buzmakov <buzmakov@gmail.com>

ENV TZ=Europe/Moscow

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN DEBIAN_FRONTEND=noninteractive  apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion && \
    apt-get clean && \
    apt-get autoremove -y && apt-get autoclean -y &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /rbtm

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

COPY environment.yml /rbtm/

RUN /opt/conda/bin/conda env create -f environment.yml  && /opt/conda/bin/conda clean -y --all 

COPY . /rbtm
RUN echo "MONGODB_URI = 'mongodb://web_database_1:27017'" > /rbtm/recon/conf.py
VOLUME /fast
VOLUME /storage

# RUN useradd -ms /bin/bash makov
# USER makov
CMD bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate xrecon && cd /rbtm/recon/ && python tomo_worker.py"
