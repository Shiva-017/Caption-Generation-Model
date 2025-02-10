# utils/monitoring.py

import psutil
import GPUtil
import time
import logging

def setup_logger(log_file='resource_monitor.log'):
    """
    Set up logging configuration.

    Args:
        log_file (str): File to save logs.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def monitor_resources(interval=5):
    """
    Continuously monitor CPU, RAM, and GPU usage.

    Args:
        interval (int): Time interval between logs in seconds.
    """
    setup_logger()
    logging.info("Starting resource monitoring...")
    try:
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            # RAM usage
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            # GPU usage
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
            # Log the information
            logging.info(f"CPU Usage: {cpu_percent}% | RAM Usage: {ram_percent}% | GPU Info: {gpu_info}")
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Resource monitoring stopped.")
        print("Resource monitoring stopped.")

if __name__ == "__main__":
    """
    To use this script, run it in a separate terminal window:
    
    python utils/monitoring.py
    
    To stop the monitoring, press Ctrl+C.
    """
    monitor_resources()
