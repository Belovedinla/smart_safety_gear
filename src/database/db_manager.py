import sqlite3
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_path="database/safety_logs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                employee_name TEXT,
                missing_ppe TEXT,
                risk_level TEXT,
                snapshot_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_violation(self, employee_name, missing_ppe, risk_level, snapshot_path=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO violation_logs (timestamp, employee_name, missing_ppe, risk_level, snapshot_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, employee_name, missing_ppe, risk_level, snapshot_path))
        conn.commit()
        conn.close()

    def get_recent_violations(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM violation_logs ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        records = cursor.fetchall()
        conn.close()
        return records
    
    def get_todays_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute('''
            SELECT COUNT(*) FROM violation_logs WHERE timestamp LIKE ?
        ''', (f"{today}%",))
        count = cursor.fetchone()[0]
        conn.close()
        return count
