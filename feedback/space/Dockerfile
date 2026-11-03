FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir 'huggingface-hub>=0.34,<2'
COPY feedback /app/feedback
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "feedback.server", "--host", "0.0.0.0", "--port", "7860"]
