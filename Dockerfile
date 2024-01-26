FROM python:3.11

# Allow statements and log messages to immediately appear in the Cloud Run logs
ARG TEST
ENV PYTHONUNBUFFERED True

ARG PORT
ENV PORT=$PORT

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install supervisord
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/

# Copy setup files, requirements, and scripts
COPY setup.py requirements.txt celeryworker.sh ./

# Make celeryworker.sh executable
RUN chmod +x ./celeryworker.sh

# Install dependencies
RUN pip install --upgrade pip==23.0.1
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Copy the supervisord.conf file into the container
COPY supervisord.conf /etc/supervisord.conf

RUN pip install --default-timeout=120 -e .
RUN mypy

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
