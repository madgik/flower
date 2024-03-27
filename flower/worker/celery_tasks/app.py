import os

from celery import Celery

from flower.celery_app_conf import configure_celery_app_to_use_priority_queue

rabbitmq_credentials = "user:password"
rabbitmq_socket_addr = f"172.17.0.1:{os.environ.get('RABBITMQ_PORT')}"
vhost = "user_vhost"

app = Celery(
    broker=f"pyamqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
    backend="rpc://",
    include=["flower.worker.celery_tasks.client"],
)

configure_celery_app_to_use_priority_queue(app)
