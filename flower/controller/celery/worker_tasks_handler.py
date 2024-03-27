import logging
from typing import Final

from flower.controller.celery.app import CeleryAppFactory

TASK_SIGNATURES: Final = {
    "start_flower_client": "flower.worker.celery_tasks.client.start_flower_client",
}


class WorkerTasksHandler:
    def __init__(
        self,
        port: str,
    ):
        self.port = port
        self._tasks_timeout = 60

    def _get_node_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=f"172.17.0.1:{self.port}")

    def start_flower_client(self):
        logger = get_logger()
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["start_flower_client"]
        celery_app.get(
            task_signature=task_signature,
            logger=logger,
        )


def get_logger():
    logger = logging.getLogger("FLOWER")
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - CONTROLLER - %(module)s - %(funcName)s(%(lineno)d) - FLOWER - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



