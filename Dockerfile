FROM ubuntu
LABEL maintainer "asv13@pitt.edu"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git cmake make g++ gcc wget swig python-pip pkg-config
RUN apt-get install -y libgsl-dev libfftw3-dev
RUN pip install numpy scipy matplotlib jupyter

# Installing CCL C library
RUN git clone https://github.com/LSSTDESC/CCL && cd CCL && \
    mkdir -p build && (cd build; cmake .. ; make install)

# Installing CCL Python module
RUN pip install git+git://github.com/LSSTDESC/CCL.git

ENV LD_LIBRARY_PATH /usr/local/lib
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

WORKDIR /home/CCL

CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
