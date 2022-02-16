FROM python:3.8

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get upgrade -y \
  gcc \
  g++ &&\
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip==22.0.3

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY setup.py .
RUN pip install --no-cache-dir .

# The number of gunicorn workers is selected by ops based on k8s configuration.
CMD exec gunicorn --bind :3000 --worker-class sync --threads 1 --timeout 0 seer:app
