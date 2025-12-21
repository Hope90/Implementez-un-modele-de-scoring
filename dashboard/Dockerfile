FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ .
COPY model/ /model/
COPY data/ /data/

CMD ["bash", "-c", "streamlit run dashboard/dashboard.py --server.port ${PORT} --server.address 0.0.0.0"]
