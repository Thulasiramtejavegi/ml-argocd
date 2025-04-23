# Use Python slim image as the base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files, API code, and other necessary files
COPY api.py .
COPY isolation_forest_model.joblib .
COPY scaler.joblib .

# Expose the port the app runs on
EXPOSE 8081

# Set environment variables
ENV MODEL_PATH=isolation_forest_model.joblib
ENV SCALER_PATH=scaler.joblib
ENV PORT=8081

# Command to run the application
CMD ["python", "api.py"]
