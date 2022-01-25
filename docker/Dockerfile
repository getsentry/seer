FROM python:3.8

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

COPY requirements.txt .
RUN pip install --upgrade pip==21.3.1 &&\
    pip install pystan==2.19.1.1 &&\
    pip install -r requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY main.py prophet_detector.py ./

CMD exec gunicorn --bind :$PORT --workers 4 --timeout 0 main:app
