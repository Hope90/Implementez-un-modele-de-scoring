FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ .
COPY model/ /model/
COPY data/ /data/

CMD ["sh", "-c", "streamlit run dashboard.py --server.port ${PORT} --server.address 0.0.0.0"]
