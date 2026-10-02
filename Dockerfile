FROM python:3.12-slim
WORKDIR /app
COPY . .
ENV PYTHONUNBUFFERED=1 DB_PATH=/app/state/bot.db
CMD ["python", "bot.py"]
