FROM python:2.7
LABEL maintainer "asv13@pitt.edu"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git make g++ gcc wget swig libtool autoconf
RUN pip install numpy ipython[all] scipy matplotlib

ENV GSL_TAR="gsl-2.3.tar.gz"
ENV GSL_DL="http://ftp.wayne.edu/gnu/gsl/$GSL_TAR"
ENV FFTW_TAR="fftw-3.3.6-pl2.tar.gz"
ENV FFTW_DL="http://www.fftw.org/$FFTW_TAR"

ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /gnu

RUN wget -q $GSL_DL \
    && tar zxvf $GSL_TAR \
    && rm -f $GSL_TAR \
    && cd /gnu/gsl-2.3 \
    && ./configure \
    && make -j 4 \
    && make install

WORKDIR /fftw

RUN wget -q $FFTW_DL \
    && tar zxvf $FFTW_TAR \
    && rm -f $FFTW_TAR \
    && cd /fftw/fftw-3.3.6-pl2 \
    && ./configure --enable-shared \
    && make \
    && make install

RUN cd /home \
    && git clone https://github.com/LSSTDESC/CCL.git \
    && cd /home/CCL \
    && ./configure \
    && make \
    && make install \
    && autoreconf -i \
    && python setup.py install 

WORKDIR /home/CCL

CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
