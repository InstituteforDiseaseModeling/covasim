FROM continuumio/anaconda3:latest

ENV PATH="/opt/conda/bin:${PATH}"

#RUN apt-get -y update && apt-get install -y \
#    locales curl apt-utils apt-transport-https debconf-utils gcc\
#    build-essential g++ nginx apt-utils gnupg curl libgl1-mesa-glx \
#    supervisor freetype* libxml2-dev libxmlsec1-dev\
#    && rm -rf /var/lib/apt/lists/*

RUN wget "http://security-cdn.debian.org/debian-security/pool/updates/main/o/openssl/libssl1.0.0_1.0.1t-1+deb8u12_amd64.deb"
RUN apt install ./libssl1.0.0_1.0.1t-1+deb8u12_amd64.deb

RUN export DEBIAN_FRONTEND=noninteractive
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN locale-gen

COPY docker_nginx.conf /etc/nginx/sites-enabled/default

RUN python3 setup.py install
RUN python3 -m pip install gunicorn
RUN python3 -m pip install plotly_express

CMD /etc/init.d/nginx restart && supervisord -c supervisord.conf
