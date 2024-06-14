FROM golang:1.20.0 as go-build
RUN mkdir go && cd go && export GOPATH=/go && \
    go install github.com/cloudflare/cfssl/cmd/...@latest

FROM nvidia/cuda:12.3.2-base-ubuntu22.04
COPY --from=go-build /go/bin/* /bin/

# Allow statements and log messages to immediately appear in the Cloud Run logs
ARG TEST
ARG DEV
ENV PYTHONUNBUFFERED True

ARG PORT
ENV PORT=$PORT

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install Python and pip
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3-pip python3.11-dev

# Make python3.11 the default python version if necessary
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Make it available for ides that look in this directory by default.
RUN ln -s /usr/bin/python /usr/local/bin/python && \
    ln -s /usr/bin/python3 /usr/local/bin/python3

# Install libpq-dev for psycopg & git for 'sentry-sdk @ git://' in requirements.txt
RUN apt-get update && \
    apt-get install -y supervisor \
    curl \
    protobuf-compiler \
    libpq-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/

# Copy setup files, requirements, and scripts
COPY setup.py requirements.txt celeryworker.sh asyncworker.sh gunicorn.sh ./

# Make celeryworker.sh and asyncworker.sh executable
RUN chmod +x ./celeryworker.sh ./asyncworker.sh ./gunicorn.sh

# Install dependencies
RUN pip install --upgrade pip==24.0

COPY protobuf-adaptors/ protobuf-adaptors/
RUN pip wheel --default-timeout=120 ./protobuf-adaptors

COPY protos/ protos/
RUN pip install --default-timeout=120 --find-links=./ ./protos

RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

COPY certs/ certs/

# Copy the supervisord.conf file into the container
COPY supervisord.conf /etc/supervisord.conf

RUN pip install --default-timeout=120 -e .

ENV FLASK_APP=src.seer.app

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
