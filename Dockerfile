FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r backend/requirements.txt

EXPOSE 5000

CMD ["python", "backend/app.py"]
