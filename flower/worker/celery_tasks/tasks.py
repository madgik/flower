from flower.worker.celery_tasks.celery_config import celery


@celery.task
def print_hello():
    print("Hello World")
    return "Hello World"
