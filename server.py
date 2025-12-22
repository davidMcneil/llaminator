from collections import deque
from datetime import datetime
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask import request as flask_request
from types import SimpleNamespace
import base64
import cv2
import numpy as np
import ollama
import os
import queue
import ssl
import tempfile
import threading
import time


app = Flask(__name__)
app.config["SECRET_KEY"] = "robot-game-secret-key"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
)

# Configuration
MODEL = "moondream:1.8b"
# MODEL = 'gemma3:4b'
OBJECTIVE = "grab a red ball"
POSSIBLE_COMMANDS = ["forward", "turn left", "turn right", "grab", "objective complete"]
SYSTEM_PROMPT = f"""You are guiding a human to complete an objective.
Objective: {OBJECTIVE}
Possible commands: {', '.join(POSSIBLE_COMMANDS) if POSSIBLE_COMMANDS else 'any command'}
Do not reply with anything other than a possible command.
If the objective appears to be completed, respond with "objective complete"."""
# For now no system prompt just describe the image
SYSTEM_PROMPT = None
USER_PROMPT = "describe the image"

# Game state
state_lock = threading.Lock()
initial_state = SimpleNamespace(
    # Whether or not the game is running. If not running, the server should not process frames
    # and the client should not send them to the server.
    running=False,
    # The server can only handle one client at a time. If a client with a different uuid tries to
    # connect, the server should send an error to the client trying to connect and not accept
    # frames from that client.
    client_id=None,
    # If the objective is completed, the client should display this to the user.
    completed=False,
    # The current command the server is giving to the client to display and read.
    command=None,
    # A last in first out queue of frames to process. Set it to a size of 2 for now.
    frames=queue.LifoQueue(maxsize=2),  # LIFO queue (stack) with max size 2
    # The number of frames processed by the server.
    total_frames=0,
    # The total time spent processing frames by the server. This is used to calculate the FPS.
    total_processing_time=0.0,
    # The time spent processing the last frame by the server. This is used to calculate the instantaneous FPS.
    last_processing_time=0.0,
)
state = SimpleNamespace(
    running=initial_state.running,
    client_id=initial_state.client_id,
    completed=initial_state.completed,
    command=initial_state.command,
    frames=queue.LifoQueue(maxsize=2),
    total_frames=initial_state.total_frames,
    total_processing_time=initial_state.total_processing_time,
    last_processing_time=initial_state.last_processing_time,
)


def reset_state():
    """Reset state to initial values and create a new empty frames queue"""
    state.frames = queue.LifoQueue(maxsize=2)
    state.running = initial_state.running
    state.completed = initial_state.completed
    state.command = initial_state.command
    state.total_frames = initial_state.total_frames
    state.total_processing_time = initial_state.total_processing_time
    state.last_processing_time = initial_state.last_processing_time


def stats():
    """Calculate and return last_fps, average_fps, last_time, avg_time (in seconds)"""
    with state_lock:
        total_frames = state.total_frames
        total_time = state.total_processing_time
        last_time = state.last_processing_time

    if total_frames == 0:
        return 0.0, 0.0, 0.0, 0.0

    last_fps = 1.0 / last_time if last_time > 0 else 0.0
    average_fps = total_frames / total_time if total_time > 0 else 0.0
    last_time_sec = last_time
    avg_time_sec = total_time / total_frames if total_frames > 0 else 0.0

    return last_fps, average_fps, last_time_sec, avg_time_sec


def send_state():
    """Send the current state to the client (excluding state_lock and frames queue)"""
    with state_lock:
        client_id = state.client_id
        if client_id is None:
            print("send_state: client_id is None, not sending")
            return

    last_fps, average_fps, last_time, avg_time = stats()
    state_to_send = {
        "running": state.running,
        "client_id": state.client_id,
        "completed": state.completed,
        "command": state.command,
        "total_frames": state.total_frames,
        "queue_size": state.frames.qsize(),
        "last_fps": round(last_fps, 2),
        "average_fps": round(average_fps, 2),
        "last_time": round(last_time, 3),
        "avg_time": round(avg_time, 3),
    }
    print(f"send_state: Sending state to {client_id}, running={state.running}")

    socketio.emit("state", state_to_send, room=client_id)


def send_error(message, client_id=None):
    """Send an error message to the client. If client_id is not provided, uses state.client_id"""
    if client_id is None:
        with state_lock:
            client_id = state.client_id
            if client_id is None:
                return

    socketio.emit("error", {"message": message}, room=client_id)


def process_frame_and_get_command(frame_data):
    """Process a frame and return the command from LLM"""
    try:
        # Decode base64 image, data sent from client is: data:image/jpeg;base64,<base64_encoded_data>
        image_data = base64.b64decode(frame_data.split(",")[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, "Failed to decode image"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_image_path = tmp_file.name
            # Resize for faster processing
            resized_img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.imwrite(temp_image_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        try:
            # Send request to LLM
            start_time = time.time()
            response = ollama.generate(
                model=MODEL,
                prompt=USER_PROMPT,
                system=SYSTEM_PROMPT,
                images=[temp_image_path],
            )
            end_time = time.time()
            processing_time = end_time - start_time
            # print(response)

            # Update processing stats
            with state_lock:
                state.total_frames += 1
                state.total_processing_time += processing_time
                state.last_processing_time = processing_time

            command = response["response"]
            return command, None

        finally:
            # Show the image
            if os.path.exists(temp_image_path):
                img_display = cv2.imread(temp_image_path)
                if img_display is not None:
                    cv2.imshow("Robot Game - Current Frame", img_display)
                    cv2.waitKey(1)  # Non-blocking, allows other processing
                os.unlink(temp_image_path)

    except Exception as e:
        return None, str(e)


def processing_loop():
    """Process frames from the LIFO queue when running"""
    sleep_time = 0.2
    while True:
        with state_lock:
            running = state.running
            frames_queue = state.frames

        if not running:
            time.sleep(sleep_time)
            continue

        try:
            # queue operations don't need the lock, they're thread-safe
            frame_data = frames_queue.get_nowait()
        except queue.Empty:
            time.sleep(sleep_time)
            continue

        try:
            command, error = process_frame_and_get_command(frame_data)

            if error:
                print(f"Error processing frame: {error}")
                send_error(error)
                continue

            with state_lock:
                if command == "objective complete":
                    print("Objective completed")
                    state.completed = True
                    state.running = False
                state.command = command

            last_fps, average_fps, last_time, avg_time = stats()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Command: {command} | Last FPS: {last_fps:.2f} | Avg FPS: {average_fps:.2f} | Last: {last_time:.3f}s | Avg: {avg_time:.3f}s"
            )

            send_state()

        except Exception as e:
            print(f"Error in processing loop: {e}")
            time.sleep(sleep_time)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    """Handle client connection - only allow one client at a time"""
    client_session_id = flask_request.sid
    print(f"connect received from: {client_session_id}")

    with state_lock:
        reset_state()
        state.client_id = client_session_id
    print(f"client connected: {client_session_id}")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    client_session_id = flask_request.sid
    print(f"disconnect received from: {client_session_id}")

    with state_lock:
        if state.client_id == client_session_id:
            reset_state()
    print(f"Client disconnected: {client_session_id}")


# Set processing to true, reset the other state, and send state back to the client
@socketio.on("start")
def handle_start():
    """Start processing frames"""
    global state
    client_session_id = flask_request.sid
    print(f"start received from: {client_session_id}")

    with state_lock:
        if state.client_id != client_session_id:
            print(f"Client mismatch - rejecting start request")
            send_error("You are not the active client.", client_id=client_session_id)
            return

        reset_state()
        state.running = True

    print("processing started")
    send_state()


@socketio.on("stop")
def handle_stop():
    """Stop processing frames"""
    client_session_id = flask_request.sid
    print(f"stop received from: {client_session_id}")

    with state_lock:
        if state.client_id != client_session_id:
            print(f"Client mismatch - rejecting stop request")
            send_error("You are not the active client.", client_id=client_session_id)
            return

        state.running = False

    print("processing stopped")
    send_state()


@socketio.on("frame")
def handle_frame(data):
    """Handle incoming frame from client - push to LIFO queue"""
    client_session_id = flask_request.sid
    # print(f"frame received from: {client_session_id}")

    frame_data = data.get("image")
    if not frame_data:
        return

    with state_lock:
        if state.client_id != client_session_id:
            print(f"Client mismatch - rejecting frame request")
            send_error("You are not the active client.", client_id=client_session_id)
            return

        if not state.running:
            return

        # If queue is full, remove oldest (bottom) and add new (top)
        if state.frames.full():
            state.frames.get_nowait()  # Remove oldest
        state.frames.put_nowait(frame_data)  # Add new frame to top


if __name__ == "__main__":

    # Start processing thread
    processing_thread = threading.Thread(target=processing_loop, daemon=True)
    processing_thread.start()

    cert_file = os.path.join(os.path.dirname(__file__), "cert.pem")
    key_file = os.path.join(os.path.dirname(__file__), "key.pem")

    print("=" * 70)
    print("llaminator - airtight. spitt'en good. portmanteau powered. robotics!")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"System Prompt: {SYSTEM_PROMPT}")
    print(f"User Prompt: {USER_PROMPT}")

    print("Server starting on https://0.0.0.0:5001")
    print("⚠️  Using self-signed certificate - browser will show security warning")
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=True,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context,
    )
