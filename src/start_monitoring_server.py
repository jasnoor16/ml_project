from ml_utils.monitoring import RegressionMonitor

import time
import socket

def check_port_open(port=8002):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0

if __name__ == "__main__":
    monitor = RegressionMonitor(port=8002)
    monitor.start_server()

    # Wait until port is actually open
    for i in range(5):
        if check_port_open():
            print("✅ Port 8002 is open!")
            break
        else:
            print("⏳ Waiting for port 8002...")
            time.sleep(2)

    print("✅ Prometheus server running — holding the process alive.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("🛑 Monitoring server stopped.")
