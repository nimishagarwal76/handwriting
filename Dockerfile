FROM python:3.6-slim

RUN apt-get update
RUN pip3 install --upgrade pip
RUN apt-get install -y python3-tk
RUN apt-get install -y gtk2.0


WORKDIR /handwriting_synthesis
ADD . .

RUN pip3 install -r requirements.txt
ENTRYPOINT [ "/bin/sh" ]