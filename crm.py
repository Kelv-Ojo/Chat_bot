"""Small local CRM. One enquiry per submitted message; no external calls."""
import re
import sqlite3
from contextlib import contextmanager, closing
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ('New', 'In progress', 'Resolved')


@contextmanager
def database(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys = ON')
            yield connection


def initialise(path):
    with database(path) as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS enquiries (
            id INTEGER PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
            contact_id INTEGER NOT NULL REFERENCES contacts(id),
            message TEXT NOT NULL, reply TEXT NOT NULL, intent TEXT NOT NULL,
            engine TEXT NOT NULL, score REAL,
            needs_review INTEGER NOT NULL CHECK(needs_review IN (0,1)),
            status TEXT NOT NULL DEFAULT 'New'
                CHECK(status IN ('New','In progress','Resolved')),
            created_at TEXT NOT NULL, resolved_at TEXT
        );
        ''')


def capture(path, name, email, message, result, request_id):
    name, email, message = name.strip(), email.strip().lower(), message.strip()
    if not 1 <= len(name) <= 100:
        raise ValueError('Enter a name of 1–100 characters.')
    if len(email) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        raise ValueError('Enter a valid email address.')
    if not 1 <= len(message) <= 2000:
        raise ValueError('Enter an enquiry of 1–2,000 characters.')
    if not request_id:
        raise ValueError('A submission identifier is required.')
    now = datetime.now(timezone.utc).isoformat()
    with database(path) as db:
        previous = db.execute('SELECT id FROM enquiries WHERE request_id=?', (request_id,)).fetchone()
        if previous:
            return previous['id']
        db.execute('INSERT INTO contacts(email,name,created_at) VALUES(?,?,?) '
                   'ON CONFLICT(email) DO UPDATE SET name=excluded.name', (email, name, now))
        contact_id = db.execute('SELECT id FROM contacts WHERE email=?', (email,)).fetchone()['id']
        cursor = db.execute('''INSERT INTO enquiries
            (request_id,contact_id,message,reply,intent,engine,score,needs_review,created_at)
            VALUES(?,?,?,?,?,?,?,?,?)''',
            (request_id, contact_id, message, result.reply, result.intent,
             result.engine, result.score, int(result.needs_review), now))
        return cursor.lastrowid


def rows(path):
    with database(path) as db:
        return [dict(row) for row in db.execute('''SELECT e.*, c.name, c.email
            FROM enquiries e JOIN contacts c ON c.id=e.contact_id
            ORDER BY e.id DESC''')]


def contacts(path):
    with database(path) as db:
        return [dict(row) for row in db.execute('''SELECT c.id,c.name,c.email,
            COUNT(e.id) AS enquiries FROM contacts c LEFT JOIN enquiries e
            ON c.id=e.contact_id GROUP BY c.id ORDER BY c.id''')]


def set_status(path, enquiry_id, status):
    if status not in STATUSES:
        raise ValueError('Choose a recognised status.')
    resolved = datetime.now(timezone.utc).isoformat() if status == 'Resolved' else None
    with database(path) as db:
        cursor = db.execute('UPDATE enquiries SET status=?, resolved_at=? WHERE id=?',
                            (status, resolved, enquiry_id))
        if cursor.rowcount != 1:
            raise ValueError('Enquiry not found.')


def metrics(path):
    with database(path) as db:
        row = db.execute('''SELECT COUNT(*) AS enquiries,
            COUNT(DISTINCT contact_id) AS customers,
            COALESCE(SUM(needs_review),0) AS needs_review,
            COALESCE(SUM(status != 'Resolved'),0) AS open,
            COALESCE(SUM(status = 'Resolved'),0) AS resolved FROM enquiries''').fetchone()
        result = dict(row)
    result['review_rate'] = result['needs_review'] / result['enquiries'] if result['enquiries'] else None
    return result

