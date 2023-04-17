FROM python:3.8

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY setup.py requirements.txt ./
RUN pip install --upgrade pip==23.1

COPY src/ src/
RUN pip install --default-timeout=120 .

# The number of gunicorn workers is selected by ops based on k8s configuration.
CMD exec gunicorn --bind :$PORT --worker-class sync --threads 1 --timeout 0 src.seer.seer:app