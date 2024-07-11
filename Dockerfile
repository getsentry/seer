FROM pytorch/torchserve:latest

# Allow statements and log messages to immediately appear in the Cloud Run logs
ARG TEST
ARG DEV
ENV PYTHONUNBUFFERED True

ARG PORT
ENV PORT=$PORT

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/

# Debugging: List contents of models directory
RUN ls -l /app/models

RUN ls -l /app/models/issue_severity_v0


# Copy setup files, requirements, and scripts
COPY setup.py requirements.txt ./

# Switch to root user to install git
USER root
RUN apt-get update --allow-releaseinfo-change && apt-get install -y git

# Install pip and dependencies as root
RUN pip install --upgrade pip==24.0
RUN pip install -r requirements.txt --no-cache-dir

# Copy source code
COPY src/ src/
COPY pyproject.toml .

RUN pip install --default-timeout=120 -e . --no-cache-dir

# Copy TorchServe configuration files
COPY torchserve/config.properties config.properties
COPY torchserve/model-config.json model-config.json

# Debugging: List contents of app directory
RUN ls -l /app

# Expose ports for TorchServe
EXPOSE 8080 8081

# Start TorchServe
CMD ["torchserve", "--start", "--ncs", "--ts-config", "config.properties", "--model-store", "/app/models", "--models", "issue_severity_v0=model-config.json"]
