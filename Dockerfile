FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY kubernetes/deployment.yaml .
COPY isolation_forest_model.joblib .
COPY scaler.joblib .
EXPOSE 8081
ENV MODEL_PATH=isolation_forest_model.joblib
ENV SCALER_PATH=scaler.joblib
ENV PORT=8081
CMD ["python", "src/api.py"]
