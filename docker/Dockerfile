FROM continuumio/miniconda3:latest

ENV PATH="/opt/conda/bin:${PATH}"

RUN apt-get -y update && apt-get install -y nginx supervisor curl

# conda has its own gcc. Install it so we can compile required packages.
RUN conda install gcc_linux-64

RUN mkdir /app && \
    conda install twisted

# Add requirements.txt first and install to maximize chances of hitting docker cache
ADD ./requirements.txt /app
# When installing packages that need compilation, run pip in a bash shell with
# the conda environment activated. This allows it to find the conda-installed gcc.
RUN /bin/bash -c ". activate && python3 -m pip install -r /app/requirements.txt"

ADD . /app
WORKDIR /app

COPY docker/docker_nginx.conf /etc/nginx/sites-enabled/default

RUN python3 -m pip install .
# bytcode compile python libraries to improve startup times
RUN python3 -m compileall $(python3 -c "import covasim,os;print(os.path.dirname(covasim.__file__))" | tail -n 1)
RUN cd data && ./run_scrapers

CMD /etc/init.d/nginx restart && supervisord -c docker/supervisord.conf
