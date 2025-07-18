from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import io
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import ffmpeg
import tempfile
import os

# Import the ask_gemini function from your module
from gemini_module import ask_gemini

# --- Model Initialization ---
# Load the Whisper model once when the server starts
# Using "base.en" for a good balance of speed and accuracy.
# Using "cpu" and "int8" for wider compatibility and lower resource usage.
print("Loading Whisper model...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
print("Whisper model loaded.")

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!Jarvis'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Dictionary to store audio buffers for each client session
audio_buffers = {}



# --- SocketIO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    session_id = request.sid
    print(f'Client connected: {session_id}')
    # Create a new in-memory buffer for the client
    audio_buffers[session_id] = io.BytesIO()
    emit('response', {'data': 'Connected to the JARVIS server!'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    session_id = request.sid
    print(f'Client disconnected: {session_id}')
    # Clean up the buffer for the disconnected client
    if session_id in audio_buffers:
        del audio_buffers[session_id]


@socketio.on('chat_message')
def handle_chat_message(json):
    """Handles incoming chat messages from a client."""
    user_input = json.get('message')
    print(f"Received message: {user_input}")
    
    if not user_input:
        emit('error', {'error': 'No message provided.'})
        return

    reply = ask_gemini(user_input)
    
    if "[Gemini Error]" in reply:
        emit('error', {'error': reply})
    else:
        emit('chat_reply', {'reply': reply})

def transcribe_audio_buffer(audio_buffer):
    """Helper function to transcribe audio from a buffer."""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(audio_buffer.getvalue())
            temp_file_path = temp_file.name

        out, err = (
            ffmpeg
            .input(temp_file_path)
            .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        audio_np = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, info = whisper_model.transcribe(audio_np, beam_size=5)
        transcript = "".join(segment.text for segment in segments)
        return transcript.strip()

    except ffmpeg.Error as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
        return None # Indicate failure
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@socketio.on('audio_stream')
def handle_audio_stream(audio_data):
    """Receives audio stream data and provides a partial transcript."""
    session_id = request.sid
    if session_id in audio_buffers:
        audio_buffers[session_id].write(audio_data)
        
        # For real-time feedback, we transcribe the whole buffer so far.
        # This is inefficient but provides a simple way to get partial transcripts.
        transcript = transcribe_audio_buffer(audio_buffers[session_id])
        if transcript is not None:
            emit('partial_transcript', {'transcript': transcript})

@socketio.on('stop_stream')
def handle_stop_stream():
    """Finalizes transcription when the client signals the stream has stopped."""
    session_id = request.sid
    print(f"Stop stream signal received from {session_id}. Finalizing transcription...")

    if session_id in audio_buffers and audio_buffers[session_id].getbuffer().nbytes > 0:
        final_transcript = transcribe_audio_buffer(audio_buffers[session_id])
        if final_transcript is not None:
            print(f"Final transcript: {final_transcript}")
            emit('transcript_ready', {'transcript': final_transcript})
        else:
            emit('error', {'error': 'Final transcription failed.'})
        
        # Reset the buffer for the next stream
        audio_buffers[session_id].seek(0)
        audio_buffers[session_id].truncate(0)
    else:
        emit('transcript_ready', {'transcript': ''}) # Send empty if no audio



if __name__ == '__main__':
    print("Starting JARVIS WebSocket server on port 5002...")
    # allow_unsafe_werkzeug=True is required for debug mode with SocketIO
    socketio.run(app, debug=True, port=5002, allow_unsafe_werkzeug=True)
