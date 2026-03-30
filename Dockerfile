FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 7331
CMD ["llmregress", "serve"]
