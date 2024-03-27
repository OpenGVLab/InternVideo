import os
import shutil
import socket


def has_slurm():
    """determine the system has slurm or not
    Returns: True if has else False.

    """
    return shutil.which("sbatch") is not None

def random_port():
    """random a unused port
    Returns: str

    """
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def runcmd(cmd):
    """run command

    Args:
        cmd (str): The command to run

    """
    os.system(cmd)
