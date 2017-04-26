FROM python:2.7
LABEL maintainer "asv13@pitt.edu"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git make g++ gcc wget swig libtool autoconf
RUN pip install numpy ipython[all] scipy matplotlib

ENV GSL_TAR="gsl-2.3.tar.gz"
ENV GSL_DL="http://ftp.wayne.edu/gnu/gsl/$GSL_TAR"

ENV LD_LIBRARY_PATH=/usr/local/lib

ENV GITUSER=GITUSERNAMEHERE
ENV GITPASS=GITPASSWORDHERE

WORKDIR /gnu

RUN wget -q $GSL_DL \
    && tar zxvf $GSL_TAR \
    && rm -f $GSL_TAR \
    && cd /gnu/gsl-2.3 \
    && ./configure \
    && make -j 4 \
    && make install

RUN cd /home \
    && git clone https://$GITUSER:$GITPASS@github.com/LSSTDESC/CCL.git \
    && cd /home/CCL \
    && ./configure \
    && make \
    && make install \
    && autoreconf -i \
    && python setup.py install \
    && python setup.py install

WORKDIR /home/CCL

CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
