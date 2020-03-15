FROM continuumio/anaconda3:latest

ENV PATH="/opt/conda/bin:${PATH}"

RUN apt-get -y update && apt-get install -y nginx supervisor

RUN conda install twisted python-Levenshtein

ADD . /app
WORKDIR /app

COPY docker_nginx.conf /etc/nginx/sites-enabled/default

RUN python3 setup.py install
RUN python3 -m pip install gunicorn
RUN python3 -m pip install plotly_express

CMD /etc/init.d/nginx restart && supervisord -c supervisord.conf
