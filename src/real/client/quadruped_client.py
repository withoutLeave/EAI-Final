import asyncio
import websockets
import json
import time
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
import os
from dotenv import load_dotenv
load_dotenv("/home/unitree/Assignment4/hw4.env")

class QuadrupedClient:
    def __init__(self):
        host = os.getenv('SERVER_HOST')
        port = os.getenv('SERVER_PORT')
        print(f"Connecting to server at {host}:{port}...")
        self.server_uri = f"ws://{host}:{port}/quadruped"
        self.websocket = None
        self.running = False
        self.reconnect_delay = 1  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 30  # Maximum reconnect delay in seconds

        # Initialize robot control
        ChannelFactoryInitialize(0, "eth0")
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

    async def is_connected(self):
        """Check connection status"""
        return self.websocket is not None and not self.websocket.closed

    async def connect(self):
        """Connect to WebSocket server with exponential backoff"""
        while True:
            try:
                print(f"Connecting to {self.server_uri}...")
                self.websocket = await websockets.connect(
                    self.server_uri,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong response
                    close_timeout=1
                )
                self.running = True
                self.reconnect_delay = 1  # Reset reconnect delay after successful connection
                print("Successfully connected to server!")
                return True
            except Exception as e:
                print(f"Connection failed: {str(e)}")
                print(f"Retrying in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                # Exponential backoff with max limit
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def safe_close(self):
        """Safely close the connection"""
        try:
            if self.websocket is not None and not self.websocket.closed:
                await self.websocket.close()
        except Exception as e:
            print(f"Error closing connection: {str(e)}")
        finally:
            self.websocket = None
            self.running = False

    async def maintain_connection(self):
        """Maintain persistent connection to server"""
        while True:
            if not await self.is_connected():
                await self.connect()
            await asyncio.sleep(1)  # Check connection periodically

    async def receive_commands(self):
        """Continuously receive and process commands"""
        while True:
            try:
                if not await self.is_connected():
                    print("Connection lost, waiting to reconnect...")
                    await asyncio.sleep(1)
                    continue

                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1)
                    print(f"Received message: {message[:100]}...")

                    try:
                        data = json.loads(message)
                        if isinstance(data, str):
                            data = json.loads(data)
                        await self.process_command(data)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON: {str(e)}")
                        await self.send_acknowledgment(
                            "error", f"Invalid JSON: {str(e)}"
                        )

                except asyncio.TimeoutError:
                    # Timeout is normal, just continue listening
                    continue
                except (ConnectionClosedError, ConnectionClosedOK) as e:
                    print(f"Connection closed: {str(e)}")
                    await self.safe_close()
                    continue
                except Exception as e:
                    print(f"Receive error: {str(e)}")
                    await self.safe_close()
                    continue

            except Exception as e:
                print(f"Unexpected error in receive loop: {str(e)}")
                await self.safe_close()
                await asyncio.sleep(1)
                continue

    async def process_command(self, data):
        """Process incoming command"""
        try:
            print(f"Processing command: {json.dumps(data, indent=2)}")
            if data["type"] == "quadruped_control":
                await self.handle_movement_command(data)
            else:
                error_msg = f"Unknown command type: {data['type']}"
                print(error_msg)
                await self.send_acknowledgment("error", error_msg)

        except Exception as e:
            print(f"Command processing failed: {str(e)}")
            await self.send_acknowledgment("error", str(e))

    async def handle_movement_command(self, command):
        """Handle movement command"""
        try:
            x = max(-2.0, min(2.0, float(command["x_speed"])))
            y = max(-2.0, min(2.0, float(command["y_speed"])))
            yaw = max(-3.14, min(3.14, float(command["yaw_speed"])))

            print(f"Executing movement: X={x:.2f}, Y={y:.2f}, Yaw={yaw:.2f}")
            self.sport_client.Move(x, y, yaw)
            time.sleep(1.0)

            await self.send_acknowledgment("success", "Movement executed")

        except Exception as e:
            print(f"Movement failed: {str(e)}")
            await self.send_acknowledgment("error", str(e))

    async def send_acknowledgment(self, status, message):
        """Send acknowledgment back to server"""
        if await self.is_connected():
            try:
                ack = {
                    "type": "command_ack",
                    "status": status,
                    "message": message,
                    "timestamp": time.time(),
                }
                await self.websocket.send(json.dumps(ack))
            except Exception as e:
                print(f"Failed to send ack: {str(e)}")


async def main():
    client = QuadrupedClient()

    # Create tasks for both connection maintenance and command processing
    connection_task = asyncio.create_task(client.maintain_connection())
    command_task = asyncio.create_task(client.receive_commands())

    try:
        # Run both tasks concurrently
        await asyncio.gather(connection_task, command_task)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
    finally:
        # Cancel tasks and clean up
        connection_task.cancel()
        command_task.cancel()
        await client.safe_close()
        print("Controller stopped.")


if __name__ == "__main__":
    asyncio.run(main())