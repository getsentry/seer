FROM python:3.11

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy setup files and requirements
COPY setup.py requirements.txt ./

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/

# Copy source code
COPY src/ src/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --default-timeout=120 .

# The number of gunicorn workers is selected by ops based on k8s configuration.
CMD exec gunicorn --bind :9090 --worker-class sync --threads 1 --timeout 0 src.seer.seer:app