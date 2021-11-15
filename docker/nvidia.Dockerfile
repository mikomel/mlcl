FROM nvcr.io/nvidia/pytorch:21.08-py3

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
