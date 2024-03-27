import json
from pathlib import Path
from time import sleep

from flask import Flask

from flower.algorithms.process_manager import FlowerProcess
from flower.controller.celery.worker_tasks_handler import WorkerTasksHandler

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

app = Flask(__name__)


@app.route('/algorithm', methods=["POST"])
def run_algorithm():
    # Start server
    flower_server = FlowerProcess("logistic_regression/server.py")
    flower_server.start()

    sleep(1)

    # Start clients
    task_handlers = [WorkerTasksHandler(port) for port in ["5670", "5671"]]
    for task_handler in task_handlers:
        task_handler.start_flower_client()

    # Wait for server to end
    flower_server.wait(timeout=20)

    # Get result
    with open(PROJECT_ROOT / "flower" / "algorithms" / "result.json", "r") as f:
        return json.load(f)


if __name__ == '__main__':
    app.run(debug=True)
