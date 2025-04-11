# Use official Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entry point
CMD ["python", "data_drift_check.py"]
