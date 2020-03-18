FROM continuumio/anaconda3:latest

ENV PATH="/opt/conda/bin:${PATH}"

RUN apt-get -y update && apt-get install -y nginx supervisor
RUN mkdir /app && \
    conda install twisted python-Levenshtein

# Add requiremments.txt first and install to maximize chances of hitting docker cache
ADD requirements.txt /app
RUN python3 -m pip install -r /app/requirements.txt

ADD . /app
WORKDIR /app

COPY docker_nginx.conf /etc/nginx/sites-enabled/default

RUN python3 -m pip install .

CMD /etc/init.d/nginx restart && supervisord -c supervisord.conf
