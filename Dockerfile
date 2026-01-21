FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 30100

CMD ["python", "-m", "tracker.track_detections", "--tcp", "--tcp-host", "0.0.0.0", "--tcp-port", "30100", "-s", "-"]
