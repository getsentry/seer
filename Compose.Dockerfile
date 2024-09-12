FROM docker:cli

ARG SBX_PROJECT
ENV SBX_PROJECT=${SBX_PROJECT}

RUN mkdir /app
WORKDIR /app

COPY .env /app/
COPY docker-compose.yml /app/
COPY docker-compose.staging.yml /app/
# Can't reset these values with overlay unfortunately
RUN grep -v 'context: .' /app/docker-compose.yml | grep -v 'build:' > /app/docker-compose.yml.2
RUN mv /app/docker-compose.yml.2 /app/docker-compose.yml

CMD ["/usr/local/bin/docker", "compose", "-f", "/app/docker-compose.yml", "-f", "docker-compose.staging.yml", "up"]
