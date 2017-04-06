FROM continuumio/anaconda
LABEL maintainer "asv13@pitt.edu"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git make g++ gcc wget swig libtool autoconf

ENV GSL_TAR="gsl-2.3.tar.gz"
ENV GSL_DL="http://ftp.wayne.edu/gnu/gsl/$GSL_TAR"

ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /gnu

RUN wget -q $GSL_DL \
    && tar zxvf $GSL_TAR \
    && rm -f $GSL_TAR

WORKDIR /gnu/gsl-2.3

RUN ./configure \
    && make -j 4 \
    && make install

WORKDIR /home
RUN git clone https://<GITUSERHERE>:<GITPASSHERE>@github.com/LSSTDESC/CCL.git

WORKDIR /home/CCL

RUN ./configure \ 
    && make \
    && make install \
    && autoreconf -i \
    && python setup.py install \
    && python setup.py install

CMD jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
