FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install -y libglib2.0-0

RUN mkdir -p /app/datasets
RUN mkdir -p /app/logs

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY .env /app/.env
COPY config /app/config
COPY mlcl /app/mlcl

WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:."
ENTRYPOINT ["python"]
