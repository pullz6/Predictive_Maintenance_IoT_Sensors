FROM python:3.12.11

COPY ./src

WORKDIR ./src

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "model.py"]