FROM python:2.7
LABEL maintainer "francois.lanusse@gmail.com"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git cmake make g++ gcc wget swig
RUN pip install numpy ipython[all] scipy matplotlib
RUN pip install git+git://github.com/EiffL/CCL.git

WORKDIR /home/CCL

CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
