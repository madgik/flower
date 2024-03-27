import subprocess
from pathlib import Path

ALGORITHMS_ROOT = Path(__file__).parent


class FlowerProcess:
    def __init__(self, file, parameters=None):
        if parameters is None:
            parameters = []
        self.file = file
        self.parameters = parameters
        self.proc = None

    def start(self):
        if self.proc is not None:
            raise RuntimeError("Process already started!")

        flower_executable = (ALGORITHMS_ROOT / self.file)
        command = ["poetry", "run", "python", flower_executable, *self.parameters]
        print(f"Executing command: {command}")
        self.proc = subprocess.Popen(command)

    def wait(self, timeout):
        if self.proc is None:
            raise RuntimeError("Process is not started.")

        self.proc.communicate(timeout=timeout)
