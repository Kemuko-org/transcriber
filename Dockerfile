FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY transcription_api.py .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "transcription_api.py"]
