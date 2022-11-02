FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]