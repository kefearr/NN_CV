# Обучение модели

Обучения модели выполняет скрипт train.py, для запуска которого необходимо:

1. Создать в текущей директории виртуальное окружение названием .venv:

-     python3 -m venv .venv

2. Активировать созданное виртуальное окружение:

-     source .venv/bin/activate

3. Установить зависимости из файла requirements.txt

-    pip install -r requirements.txt

4. Подготовить датасет:

- Скачать датасет со страницы Kaggle [https://www.kaggle.com/datasets/nitishabharathi/scene-classification]
- Извлечь архив в папку data2 в директорию с проектом

5. Запустить скрипт
    - python3 train.
после выполнения в корне проекта будут сохранены 3 модели. обученные модели можно скачать с гугл диска по ссылке: [https://drive.google.com/drive/folders/1tCOyrdOzyjbZZbNSuS3l973KmW8ZQq3k?usp=sharing]

# Запуск inference

1. обучение или скачивание моделей.
2. Перемещение моделей в директорию с проектом
3. Запустить контрейнер: 
   - sudo docker-compose up --build
4. Посмотреть логи:
   - sudo docker logs lab2-gorodov

