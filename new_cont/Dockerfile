FROM python:3.9

RUN mkdir /app
RUN touch /app/log.txt
RUN echo "Директория создана..." >> /app/log.txt

RUN echo "Запускаю команду..." >> /app/log.txt 
    

RUN echo "Копирую файлы..." >> /app/log.txt

COPY . /app/

RUN echo "Устанавливаю зависимости..." >> /app/log.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN echo "Создаю рабочую директорию..." >> /app/log.txt

WORKDIR /app

RUN echo "Запускаю выбор скрипта..." >> /app/log.txt

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

