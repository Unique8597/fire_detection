FROM python:3-alpine3.20

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit","Home.py"]

EXPOSE 8000
