# app.py
from flask import Flask

from flower.central_node.celery.node_tasks_handler import NodeAlgorithmTasksHandler

app = Flask(__name__)


@app.route('/run_algorithm')
def run_algorithm():
    task_handlers = [NodeAlgorithmTasksHandler(port) for port in ["5671", "5672"]]
    for task_handler in task_handlers:
        task_handler.print_hello()
    return "Hello World task has been enqueued!"


if __name__ == '__main__':
    app.run(debug=True)
