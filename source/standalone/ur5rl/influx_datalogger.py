import queue
import time
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
import threading


class InfluxDataLogger:
    def __init__(self, influx_url, org, run_info, action_scaling):
        """
        Initializes the DataLogger with InfluxDB connection details.
        """
        token = "wjdck8n2clopm1lu5LA3Pcnbby4n93GtcfO8UAmIU5Lj-BqgF_uCfqBm_mQ0uvlbk8yDuGU4JftJ9nvIkMj-Ig=="
        self.client = InfluxDBClient(url=influx_url, token=token, org=org)
        self.write_api = self.client.write_api()
        self.run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Unique ID for each run
        )
        self.run_info = run_info
        self.action_scaling = action_scaling

        self.queue = queue.Queue()  # Thread-safe queue for logging data
        self.stop_event = threading.Event()

        # Start the background logging thread
        self.logging_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.logging_thread.start()

    def log_joint_positions(
        self,
        joint_names: list[str],
        sim_positions: list,
        real_positions: list | None,
        bucket: str,
    ):
        """
        Adds joint positions to the logging queue for asynchronous processing.
        """
        data = {
            "joint_names": joint_names,
            "sim_positions": sim_positions,
            "real_positions": real_positions,
            "bucket": bucket,
        }
        self.queue.put(data)

    def _process_queue(self):
        """
        Background thread to process the logging queue.
        """
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Fetch data from the queue with a timeout
                data = self.queue.get(timeout=0.1)
                joint_names = data["joint_names"]
                sim_positions = data["sim_positions"]
                real_positions = data["real_positions"]
                bucket = data["bucket"]

                # Create points for each joint
                points = []
                # Log SIM and REAL joint positions if available
                if real_positions is not None:
                    for joint, sim, real in zip(
                        joint_names, sim_positions, real_positions
                    ):
                        point = (
                            Point("joint_data")
                            .tag("joint", joint)
                            .tag("run_id", self.run_id)
                            .field("sim", sim)
                            .field("real", real)
                            .field("difference", sim - real)
                            .field("run_info", self.run_info)
                            .field("action_scaling", self.action_scaling)
                            .time(datetime.utcnow())
                        )
                        points.append(point)
                # Log only SIM joint positions if REAL positions are not available
                else:
                    for joint, sim in zip(joint_names, sim_positions):
                        point = (
                            Point("joint_data")
                            .tag("joint", joint)
                            .tag("run_id", self.run_id)
                            .field("sim", sim)
                            .field("run_info", self.run_info)
                            .field("action_scaling", self.action_scaling)
                            .time(datetime.utcnow())
                        )
                        points.append(point)

                # Write to InfluxDB
                self.write_api.write(bucket=bucket, record=points)
            except queue.Empty:
                continue  # If the queue is empty, keep waiting

    def close(self):
        """
        Stops the logging thread and closes the InfluxDB client.
        """
        self.stop_event.set()
        self.logging_thread.join()  # Wait for the logging thread to finish
        self.client.close()
