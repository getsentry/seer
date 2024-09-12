FROM docker:cli

ARG SBX_PROJECT
ENV SBX_PROJECT=${SBX_PROJECT}

RUN mkdir /app
WORKDIR /app

COPY .env /app/
COPY docker-compose.yml /app/
COPY docker-compose.staging.yml /app/

CMD ["/usr/local/bin/docker", "compose", "-f", "/app/docker-compose.yml", "-f", "docker-compose.staging.yml", "up"]
