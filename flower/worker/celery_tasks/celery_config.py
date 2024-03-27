from celery import Celery


def make_celery():
    ip = "172.17.0.1"
    port = "5672"
    user = "user"
    password = "password"
    vhost = "user_vhost"

    rabbitmq_credentials = user + ":" + password
    rabbitmq_socket_addr = ip + ":" + str(port)
    return Celery(
        "flower.worker.celery_tasks",
        broker=f"pyamqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
        backend="rpc://",
        include=[
            "flower.worker.celery_tasks.tasks",
        ],
    )


celery = make_celery()
