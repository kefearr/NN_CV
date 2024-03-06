# Обучение модели

Для обучения использовался Google Colab, так как там удобно работать с датасетами. Также необходимо будет скачать датасет со страницы Kaggle на Google Disk и подключить его к блокноту.
[Ссылка на датасет](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition/)

После выполнения блокнота в корне блокнота будет сохранен файл `model.pth`, который необходимо скачать и поместить в папку model в корне данного проекта

[Ссылка на модели](https://drive.google.com/file/d/1P6jlKbrvv2kS1Z240Ebuk-B8UfQzh70F/view?usp=sharing)
В данном репозитории находятся уже обученные веса модели. Метрики после обучения:

`Train Loss: 1.0297 - Train Acc: 0.9756 - Test Loss: 0.7453 - Test Acc:0.9908`

# Запуск inference
Необходимо распаковать архив c датасетом в папку `archive` в корне данного проекта
Файлы должны лежать так:
- archive
  - test
  - train
  - sampleSubmission.csv

Необходимо выполнить команду `docker compose up --build` в корне данного проекта

Результат выполнения программы будет находиться в папке `output`

