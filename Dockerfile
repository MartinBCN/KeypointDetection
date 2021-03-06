FROM python:3.8

ENV DATA_PATH=/data
ENV FIG_PATH=/figures
ENV MODEL_PATH=/models

COPY requirements.txt .

RUN pip install -r /requirements.txt

COPY ./data $DATA_PATH
COPY ./models $MODEL_PATH

RUN mkdir /app
COPY ./src /app
WORKDIR /app

EXPOSE 8050

CMD ["python", "dashboard.py"]
