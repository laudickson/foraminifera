# Use the official Python 3.12 base image
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && \
    apt-get install -y python3-pyqt5 && \
    apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
EXPOSE 5000