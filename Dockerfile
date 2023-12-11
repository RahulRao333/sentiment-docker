FROM python:3.9
COPY . /app
COPY templates /app/
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
CMD python app.py


