FROM python:3.6-stretch
LABEL maintainer "francois.lanusse@gmail.com"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git cmake make g++ gcc wget swig
RUN pip install numpy ipython[all] scipy matplotlib
RUN pip install git+git://github.com/EiffL/CCL.git --verbose

WORKDIR /home/CCL

CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0
