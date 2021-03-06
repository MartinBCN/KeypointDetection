FROM python:3.8

ENV DATA_PATH=/data
ENV FIG_PATH=/figures
ENV MODEL_PATH=/models


COPY requirements.txt .

RUN pip install -r /requirements.txt

ADD src .

EXPOSE 8000

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]
