FROM python:3.6-stretch
LABEL maintainer "francois.lanusse@gmail.com"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git cmake make g++ gcc wget swig
RUN apt-get install -y libgsl-dev libfftw3-dev
RUN pip install numpy ipython[all] scipy matplotlib

# Installing CCL C library
RUN git clone https://github.com/EiffL/CCL && cd CCL && \
    mkdir -p build && (cd build; cmake .. ; make install) && \
    python setup.py install

WORKDIR /home/CCL

CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
