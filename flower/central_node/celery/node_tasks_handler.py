import logging
from typing import Final
from typing import List

from flower.central_node.celery.app import CeleryAppFactory

TASK_SIGNATURES: Final = {
    "print_hello": "flower.worker.celery_tasks.tasks.print_hello",
}


class NodeAlgorithmTasksHandler:
    def __init__(
        self,
        port: str,
    ):
        self.port = port
        self._tasks_timeout = 60

    def _get_node_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=f"172.17.0.1:{self.port}")

    def print_hello(self) -> List[str]:
        logger = get_logger()
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["print_hello"]
        result = celery_app.get(
            task_signature=task_signature,
            logger=logger,
        )
        return list(result)

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



