# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# создаем базовый образ
FROM python:3.9.13-slim as base

# Устанавливаем переменные среды
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# монтирование кеша в /root/.cache/pip для ускорения последующих сборок.
# Используйте привязку к requirements.txt, чтобы избежать необходимости копировать их в
# в этот слой.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt


# копируем файлы в контейнер
COPY . .

# устанавливаем порт
EXPOSE 8000

# запуск приложения при запуске контейнера
CMD python main.py
