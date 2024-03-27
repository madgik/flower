# Run on client1
```
docker run -d -p 5670:5672 --name rabbitmq0 madgik/exareme2_rabbitmq:0.21.1
export RABBITMQ_PORT=5670 && export PYTHONPATH=/home/thanasis/flower && poetry run celery -A flower.worker.celery_tasks.app worker -l INFO
```

# Run on client2
```
docker run -d -p 5671:5672 --name rabbitmq0 madgik/exareme2_rabbitmq:0.21.1
export RABBITMQ_PORT=5671 && export PYTHONPATH=/home/thanasis/flower && poetry run celery -A flower.worker.celery_tasks.app worker -l INFO
```

# Run on central
```
export PYTHONPATH=/home/thanasis/flower && flask --app flower.controller.flask.app:app run
```