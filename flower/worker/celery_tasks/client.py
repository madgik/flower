from celery import shared_task

from flower.algorithms.process_manager import FlowerProcess


@shared_task
def start_flower_client():
    FlowerProcess("logistic_regression/client.py").start()

