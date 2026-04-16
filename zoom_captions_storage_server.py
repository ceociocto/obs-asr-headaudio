#!/usr/bin/env python3
"""
Zoom Meeting Captions Storage Server
Captures Zoom meeting captions and stores them in SQLite database

Features:
- HTTP API for caption storage
- WebSocket support for real-time caption capture
- SQLite database with meetings, captions, and participants tables
- Caption merging for same speaker within 5 seconds
- AI speaker filtering
- Environment variable configuration from .env file

Usage: uv run python zoom_captions_storage_server.py
"""

import asyncio
import websockets
import json
import logging
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("zoom-captions-storage")

# Database configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "captions.db")

# AI speaker patterns to filter out
AI_SPEAKER_PATTERNS = [
    'echo ai', 'ai assistant', '数字人', 'digital human', 'ai', 'bot',
    'echo', 'aiko', 'mike', 'mac', 'mic'
]

# AI wake words to filter out
AI_WAKE_WORDS = [
    'hey echo', 'echo,', 'echo ', '数字人', 'ai assistant', 'ask ai', 'question for'
]


def load_env_file():
    """Load environment variables from .env file"""
    env_vars = {
        'ZOOM_SDK_KEY': '',
        'ZOOM_SDK_SECRET': '',
        'ZOOM_MEETING_NUMBER': '',
        'ZOOM_MEETING_PASSWORD': '',
        'ZOOM_BOT_NAME': 'Caption Recorder'
    }

    # Try to find .env file in multiple locations
    env_paths = [
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / 'zoom-meeting-sdk-demo' / '.env',
        Path.cwd() / '.env'
    ]

    for env_path in env_paths:
        if env_path.exists():
            log.info(f"Loading .env from: {env_path}")
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in env_vars:
                            env_vars[key] = value
                            log.debug(f"  {key} = ***" if 'SECRET' in key or 'KEY' in key else f"  {key} = {value}")
            break
    else:
        log.warning(".env file not found in any of these locations:")
        for p in env_paths:
            log.warning(f"  - {p}")

    return env_vars


# Load environment configuration
ENV_CONFIG = load_env_file()


class CaptionsDatabase:
    """SQLite database for storing Zoom meeting captions"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()

    def init_database(self):
        """Initialize database and create tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create meetings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_number TEXT NOT NULL,
                topic TEXT,
                host_name TEXT,
                started_at TEXT DEFAULT (datetime('now', 'localtime')),
                ended_at TEXT
            )
        """)

        # Create captions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS captions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                speaker TEXT,
                text TEXT NOT NULL,
                caption_type TEXT DEFAULT 'transcription',
                received_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (meeting_id) REFERENCES meetings(id)
            )
        """)

        # Create participants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                user_id TEXT,
                user_name TEXT,
                is_host INTEGER DEFAULT 0,
                joined_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (meeting_id) REFERENCES meetings(id)
            )
        """)

        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_captions_meeting_id
            ON captions(meeting_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_captions_received_at
            ON captions(received_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_participants_meeting_id
            ON participants(meeting_id)
        """)

        conn.commit()
        conn.close()
        log.info(f"Database initialized: {self.db_path}")

    def create_meeting(self, meeting_number: str, topic: str = None, host_name: str = None) -> int:
        """Create a new meeting record"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO meetings (meeting_number, topic, host_name) VALUES (?, ?, ?)",
                (str(meeting_number), topic or f"Meeting {meeting_number}", host_name or "")
            )
            meeting_id = cursor.lastrowid
            conn.commit()
            conn.close()
            log.info(f"Created meeting {meeting_id} for {meeting_number}")
            return meeting_id

    def add_caption(self, meeting_id: int, speaker: str, text: str,
                    caption_type: str = "transcription") -> bool:
        """Add a caption with merge logic"""
        # Filter out AI speakers
        speaker_lower = speaker.lower()
        if any(pattern in speaker_lower for pattern in AI_SPEAKER_PATTERNS):
            log.debug(f"Filtered AI speaker: {speaker}")
            return False

        # Filter out AI wake words
        text_lower = text.lower()
        if any(word in text_lower for word in AI_WAKE_WORDS):
            log.debug(f"Filtered AI wake word in text: {text[:50]}...")
            return False

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if we should merge with the last caption
            cursor.execute("""
                SELECT id, text, received_at
                FROM captions
                WHERE meeting_id = ?
                ORDER BY received_at DESC
                LIMIT 1
            """, (meeting_id,))

            last_row = cursor.fetchone()
            merge_threshold = 5000  # 5 seconds

            should_merge = False
            last_caption_id = None

            if last_row:
                last_id, last_text, last_received = last_row
                last_time = datetime.strptime(last_received, "%Y-%m-%d %H:%M:%S")
                time_diff = (datetime.now() - last_time).total_seconds() * 1000

                # Merge if same speaker and within threshold
                cursor.execute("SELECT speaker FROM captions WHERE id = ?", (last_id,))
                last_speaker = cursor.fetchone()[0]

                if last_speaker == speaker and time_diff < merge_threshold:
                    should_merge = True
                    last_caption_id = last_id

            if should_merge and last_caption_id:
                # Merge with existing caption
                cursor.execute(
                    "UPDATE captions SET text = text || ? WHERE id = ?",
                    (text, last_caption_id)
                )
                log.debug(f"Merged caption for speaker '{speaker}'")
            else:
                # Insert new caption
                cursor.execute(
                    "INSERT INTO captions (meeting_id, speaker, text, caption_type) VALUES (?, ?, ?, ?)",
                    (meeting_id, speaker, text, caption_type)
                )

            conn.commit()
            conn.close()
            return True

    def add_participant(self, meeting_id: int, user_id: str, user_name: str, is_host: bool = False):
        """Add a participant to the meeting"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO participants (meeting_id, user_id, user_name, is_host) VALUES (?, ?, ?, ?)",
                (meeting_id, user_id or "", user_name or "", 1 if is_host else 0)
            )
            conn.commit()
            conn.close()

    def end_meeting(self, meeting_id: int):
        """Mark meeting as ended"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE meetings SET ended_at = datetime('now', 'localtime') WHERE id = ?",
                (meeting_id,)
            )
            conn.commit()
            conn.close()

    def get_meeting_captions(self, meeting_id: int) -> List[Dict]:
        """Get all captions for a meeting"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM captions WHERE meeting_id = ? ORDER BY received_at ASC",
            (meeting_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_all_meetings(self) -> List[Dict]:
        """Get all meetings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meetings ORDER BY started_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM meetings")
        meeting_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM captions")
        caption_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM participants")
        participant_count = cursor.fetchone()[0]

        conn.close()
        return {
            "meetings": meeting_count,
            "captions": caption_count,
            "participants": participant_count
        }


# Global database instance
db = CaptionsDatabase(DB_PATH)


class APIRequestHandler(BaseHTTPRequestHandler):
    """HTTP API request handler"""

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

    def send_json_response(self, data: dict, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_cors_response(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        """Handle OPTIONS request"""
        self.send_cors_response()

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/meeting':
            # Create meeting
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            meeting_id = db.create_meeting(
                data.get('meetingNumber', ''),
                data.get('topic'),
                data.get('hostName')
            )
            self.send_json_response({'meetingId': meeting_id}, 201)
            log.info(f"API: Created meeting {meeting_id}")

        elif self.path == '/api/caption':
            # Add caption
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            success = db.add_caption(
                data.get('meetingId', 0),
                data.get('speaker', ''),
                data.get('text', ''),
                data.get('captionType', 'transcription')
            )
            self.send_json_response({'success': success}, 201 if success else 200)

        elif self.path == '/api/participant':
            # Add participant
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            db.add_participant(
                data.get('meetingId', 0),
                data.get('userId', ''),
                data.get('userName', ''),
                data.get('isHost', False)
            )
            self.send_json_response({'success': True}, 201)

        else:
            self.send_json_response({'error': 'Not found'}, 404)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/config':
            # Return configuration for frontend defaults
            config = {
                'sdkKey': ENV_CONFIG['ZOOM_SDK_KEY'],
                'sdkSecret': ENV_CONFIG['ZOOM_SDK_SECRET'],
                'meetingNumber': ENV_CONFIG['ZOOM_MEETING_NUMBER'],
                'password': ENV_CONFIG['ZOOM_MEETING_PASSWORD'],
                'displayName': ENV_CONFIG['ZOOM_BOT_NAME']
            }
            self.send_json_response(config)

        elif self.path == '/api/meetings':
            meetings = db.get_all_meetings()
            self.send_json_response({'meetings': meetings})

        elif self.path.startswith('/api/captions?'):
            # Get captions for meeting
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            meeting_id = int(params.get('meetingId', [0])[0])

            captions = db.get_meeting_captions(meeting_id)
            self.send_json_response({'captions': captions})

        elif self.path == '/api/stats':
            stats = db.get_stats()
            self.send_json_response(stats)

        else:
            # Serve static files
            file_path = self.path[1:] if self.path != '/' else 'zoom-meeting-captions.html'
            if os.path.exists(file_path):
                self.send_response(200)

                # Set content type
                if file_path.endswith('.html'):
                    content_type = 'text/html'
                elif file_path.endswith('.js'):
                    content_type = 'text/javascript'
                elif file_path.endswith('.css'):
                    content_type = 'text/css'
                elif file_path.endswith('.wasm'):
                    content_type = 'application/wasm'
                else:
                    content_type = 'application/octet-stream'

                self.send_header('Content-Type', content_type)
                self.send_header('Access-Control-Allow-Origin', '*')

                # WASM files require special headers for COOP/COEP
                if file_path.endswith('.wasm'):
                    self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
                    self.send_header('Cross-Origin-Opener-Policy', 'same-origin')

                self.end_headers()
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_json_response({'error': 'File not found'}, 404)


def run_http_server(port: int = 8768):
    """Run HTTP server in separate thread"""
    server = HTTPServer(('localhost', port), APIRequestHandler)
    log.info(f"HTTP server listening on port {port}")
    server.serve_forever()


async def websocket_handler(websocket):
    """WebSocket handler for real-time caption capture"""
    log.info("WebSocket client connected")

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))

            elif msg_type == 'caption':
                # Store caption
                db.add_caption(
                    data.get('meetingId', 0),
                    data.get('speaker', ''),
                    data.get('text', ''),
                    data.get('captionType', 'transcription')
                )
                await websocket.send(json.dumps({'type': 'caption_received'}))

            elif msg_type == 'meeting':
                # Create meeting
                meeting_id = db.create_meeting(
                    data.get('meetingNumber', ''),
                    data.get('topic'),
                    data.get('hostName')
                )
                await websocket.send(json.dumps({
                    'type': 'meeting_created',
                    'meetingId': meeting_id
                }))

            elif msg_type == 'participant':
                # Add participant
                db.add_participant(
                    data.get('meetingId', 0),
                    data.get('userId', ''),
                    data.get('userName', ''),
                    data.get('isHost', False)
                )
                await websocket.send(json.dumps({'type': 'participant_added'}))

            elif msg_type == 'end_meeting':
                # End meeting
                db.end_meeting(data.get('meetingId', 0))
                await websocket.send(json.dumps({'type': 'meeting_ended'}))

            elif msg_type == 'get_captions':
                # Get captions for meeting
                captions = db.get_meeting_captions(data.get('meetingId', 0))
                await websocket.send(json.dumps({
                    'type': 'captions',
                    'captions': captions
                }))

            elif msg_type == 'get_meetings':
                # Get all meetings
                meetings = db.get_all_meetings()
                await websocket.send(json.dumps({
                    'type': 'meetings',
                    'meetings': meetings
                }))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        log.info("WebSocket client disconnected")


async def run_websocket_server(port: int = 8767):
    """Run WebSocket server"""
    async with websockets.serve(websocket_handler, "localhost", port):
        log.info(f"WebSocket server listening on port {port}")
        await asyncio.Future()


def main():
    """Main entry point"""
    log.info("=" * 60)
    log.info("Zoom Meeting Captions Storage Server")
    log.info("=" * 60)
    log.info(f"Database: {DB_PATH}")
    stats = db.get_stats()
    log.info(f"  Meetings: {stats['meetings']}")
    log.info(f"  Captions: {stats['captions']}")
    log.info(f"  Participants: {stats['participants']}")
    log.info("")
    log.info("Servers:")
    log.info("  HTTP API: http://localhost:8768")
    log.info("  WebSocket: ws://localhost:8767")
    log.info("=" * 60)

    # Start HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, args=(8768,), daemon=True)
    http_thread.start()

    # Run WebSocket server
    asyncio.run(run_websocket_server(8767))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nShutting down...")
