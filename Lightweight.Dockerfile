FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ARG TEST
ARG DEV
ENV PYTHONUNBUFFERED True

ARG PORT
ENV PORT=$PORT

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install libpq-dev for psycopg & git for 'sentry-sdk[flask] @ git://' in requirements.txt
RUN apt-get update && \
    apt-get install -y \
    supervisor \
    curl \
    libpq-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install td-grpc-bootstrap
RUN curl -L https://storage.googleapis.com/traffic-director/td-grpc-bootstrap-0.16.0.tar.gz | tar -xz && \
    mv td-grpc-bootstrap-0.16.0/td-grpc-bootstrap /usr/local/td-grpc-bootstrap && \
    rm -rf td-grpc-bootstrap-0.16.0

COPY pyproject.toml .

# Install dependencies
COPY setup.py requirements.txt ./
RUN pip install --upgrade pip==24.0
# Install wheel package before requirements
RUN pip install wheel
# pytorch without gpu
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt --no-cache-dir

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/
# Copy scripts
COPY celeryworker.sh celerybeat.sh gunicorn.sh grpcserver.sh ./
RUN chmod +x ./celeryworker.sh ./celerybeat.sh ./gunicorn.sh ./grpcserver.sh

# Copy source code
COPY src/ src/
COPY .test_durations .


# Copy the supervisord.conf file into the container
COPY supervisord.conf /etc/supervisord.conf

# Ignore dependencies, as they are already installed and docker handles the caching
# this skips annoying rebuilds where requirements would technically be met anyways.
RUN pip install --default-timeout=120 -e . --no-cache-dir --no-deps

ENV FLASK_APP=src.seer.app:start_app()

# Supports sentry releases
ARG SEER_VERSION_SHA
ENV SEER_VERSION_SHA ${SEER_VERSION_SHA}
ARG SENTRY_ENVIRONMENT=production
ENV SENTRY_ENVIRONMENT ${SENTRY_ENVIRONMENT}

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
