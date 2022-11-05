FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip --no-cache-dir

# RUN set -xe \
#     && apt-get update -y \
#     && apt-get install --upgrade pip3 --no-cache-dir

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

# CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
CMD ["gunicorn", "DjangoAPI.wsgi:application", "--bind", "0.0.0.0:8000"]