FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

EXPOSE 30100

CMD ["python", "-m", "retina_tracker.track_detections", "--tcp", "--tcp-host", "0.0.0.0", "--tcp-port", "30100", "-s", "-"]
