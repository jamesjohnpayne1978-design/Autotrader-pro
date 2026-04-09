FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
