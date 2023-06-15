FROM python:3.9.5-buster

RUN cd src/python/
RUN pip install -r requirements.txt
RUN pip install -e .