import os

from celery import Celery


def get_celery_app(user: str, password: str, socket_addr: str, vhost: str) -> Celery:
    broker = f"pyamqp://{user}:{password}@{socket_addr}/{vhost}"
    celery_app = Celery(broker=broker, backend="rpc://",include=[
            "flower.worker.celery_tasks.tasks",
        ],)

    # connection pool disabled
    # connections are established and closed for every use
    celery_app.conf.broker_pool_limit = None

    return celery_app

ip = "172.17.0.1"
port = os.environ.get('PORT')
user = "user"
password = "password"
vhost = "user_vhost"
celery = get_celery_app(user, password, f"{ip}:{port}", vhost)
