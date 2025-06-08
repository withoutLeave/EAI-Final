import asyncio
from src.real.server.robot_server import RobotServer
from threading import Thread
import time

async def main():
    server = RobotServer()
    ws_thread = Thread(target=server.start, daemon=True)
    ws_thread.start()

    while True:
        if server.check_active_connections():
            print("Server is running and has active connections.")
            break
        else:
            print("Waiting for active connections...")
            await asyncio.sleep(1)
    try:
        for i in range(3):
            try:
                server.send_quadruped_command(0.5, 0.2, 0.0)
            except ConnectionError:
                print("No active connections. Retrying...")
            time.sleep(1.0)
        server.send_quadruped_command(0.0, 0.0, 0.0)
        time.sleep(1.0)
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
