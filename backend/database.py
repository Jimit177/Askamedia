import sqlite3
from datetime import datetime
import os

# === Path to your SQLite DB ===
DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

# === Create All Required Tables ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Chat sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TEXT
    )
    """)

    # Chat messages
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        sender TEXT,
        message TEXT,
        timestamp TEXT,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """)

    # User accounts (login/registration)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()

# === Create a New Session ===
def create_session(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("INSERT INTO sessions (name, created_at) VALUES (?, ?)", (name, now))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

# === Save Message to a Session ===
def save_message(session_id, sender, message):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("INSERT INTO messages (session_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
                   (session_id, sender, message, now))
    conn.commit()
    conn.close()

# === Fetch All Chat Sessions ===
def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM sessions ORDER BY id DESC")
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# === Fetch Messages by Session ID ===
def get_session_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sender, message, timestamp FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

# === Register a New User ===
def register_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        conn.close()
        return True, "Registered successfully."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Email already exists."

# === Validate User Login ===
def validate_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None
