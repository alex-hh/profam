# Description: This script automatically monitors and syncs offline Wandb runs to the cloud
# License: MIT License

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

# Configure logging
LOG_FILE = "monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class EventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".wandb"):
            file_name = os.path.basename(event.src_path)
            file_path = os.path.abspath(event.src_path)
            logger.info(f"File modified: {file_name}")
            self.sync_folder(os.path.dirname(file_path))

    def sync_folder(self, folder_path):
        wandb_cmd = shutil.which("wandb")
        if not wandb_cmd:
            raise FileNotFoundError(
                "Could not find 'wandb' command in PATH or at ~/.local/bin/wandb"
            )
        command = f"{wandb_cmd} sync {folder_path}"
        logger.info(f"Running command: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
            logger.info(f"Command executed successfully for {folder_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while executing command for {folder_path}: {e}")


def monitor_directory(path_to_monitor):
    event_handler = EventHandler()
    observer = PollingObserver()
    observer.schedule(event_handler, path=path_to_monitor, recursive=True)

    logger.info(f"Monitoring folder: {path_to_monitor}")
    observer.start()

    try:
        while True:
            time.sleep(180)
    except KeyboardInterrupt:
        logger.info("Stopping observer due to keyboard interrupt")
        observer.stop()
    observer.join()
    logger.info("Observer stopped")


if __name__ == "__main__":
    path_to_monitor = "/p/project1/hai_1116/ProFam/logs/wandb"  # Replace with the path you want to monitor
    if not os.path.exists(path_to_monitor):
        logger.error(f"Directory to monitor does not exist: {path_to_monitor}")
    else:
        monitor_directory(path_to_monitor)
