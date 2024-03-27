# app.py
from flask import Flask
from flower.worker.celery_tasks.tasks import print_hello

app = Flask(__name__)


@app.route('/run_algorithm')
def run_algorithm():
    print_hello.delay()
    return "Hello World task has been enqueued!"


if __name__ == '__main__':
    app.run(debug=True)
