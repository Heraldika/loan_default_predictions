FROM python

ENV PYTHONUNBUFFERED True

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT -k uvicorn.workers.UvicornWorker  --workers 1 --threads 4 --timeout 0 lgbm_api:app

