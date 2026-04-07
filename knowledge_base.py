#!/usr/bin/env python3
"""
Local Knowledge Base Module for Meeting Assistant
RAG pipeline: STT text → retrieve context → LLM generate → answer for TTS

Uses SQLite FTS5 for retrieval and local oMLX LLM for generation.
"""

import re
import sqlite3
import json
import logging
import httpx
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default config
DEFAULT_DB_PATH = str(Path(__file__).parent / "knowledge.db")
DEFAULT_LLM_URL = "http://127.0.0.1:12345/v1"
DEFAULT_LLM_KEY = "1234"
DEFAULT_MODEL = "Qwen3.5-2B-MLX-8bit"


class KnowledgeBase:
    """Local knowledge base with RAG capabilities."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        llm_url: str = DEFAULT_LLM_URL,
        llm_api_key: str = DEFAULT_LLM_KEY,
        model: str = DEFAULT_MODEL,
    ):
        self.db_path = db_path
        self.llm_url = llm_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.model = model
        self._init_db()

    # ── Database ───────────────────────────────────────────────

    def _init_db(self):
        """Initialize SQLite with FTS5 tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Meeting transcripts table
        c.execute("""
            CREATE TABLE IF NOT EXISTS meeting_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT 'default',
                meeting_title TEXT DEFAULT ''
            )
        """)

        # FTS5 index for meeting turns
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS meeting_turns_fts
            USING fts5(content, role, meeting_title, content='meeting_turns', content_rowid='id')
        """)

        # Sync triggers (keep FTS in sync)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS meeting_turns_ai AFTER INSERT ON meeting_turns BEGIN
                INSERT INTO meeting_turns_fts(rowid, content, role, meeting_title)
                VALUES (new.id, new.content, new.role, new.meeting_title);
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS meeting_turns_ad AFTER DELETE ON meeting_turns BEGIN
                INSERT INTO meeting_turns_fts(meeting_turns_fts, rowid, content, role, meeting_title)
                VALUES ('delete', old.id, old.content, old.role, old.meeting_title);
            END
        """)

        # Documents table (reuse existing or create new)
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT DEFAULT '',
                chunk_index INTEGER DEFAULT 0,
                content TEXT NOT NULL,
                source TEXT DEFAULT '',
                created_at TEXT DEFAULT ''
            )
        """)

        # FTS5 for documents
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_search
            USING fts5(content, filename, content='documents', content_rowid='id')
        """)

        c.execute("""
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                INSERT INTO docs_search(rowid, content, filename)
                VALUES (new.id, new.content, new.filename);
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                INSERT INTO docs_search(docs_search, rowid, content, filename)
                VALUES ('delete', old.id, old.content, old.filename);
            END
        """)

        conn.commit()
        conn.close()
        logger.info(f"Knowledge base initialized: {self.db_path}")

    # ── Ingestion ──────────────────────────────────────────────

    def add_meeting_turn(self, role: str, content: str, session_id: str = "default", meeting_title: str = ""):
        """Add a single meeting turn."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO meeting_turns (role, content, timestamp, session_id, meeting_title) VALUES (?, ?, ?, ?, ?)",
            (role, content, datetime.now().isoformat(), session_id, meeting_title),
        )
        conn.commit()
        conn.close()

    def add_meeting_transcript(self, turns: list[dict], session_id: str = "default", meeting_title: str = ""):
        """Add a full meeting transcript. turns: [{role, content}, ...]"""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        conn.executemany(
            "INSERT INTO meeting_turns (role, content, timestamp, session_id, meeting_title) VALUES (?, ?, ?, ?, ?)",
            [(t["role"], t["content"], now, session_id, meeting_title) for t in turns],
        )
        conn.commit()
        conn.close()
        logger.info(f"Added {len(turns)} turns for meeting '{meeting_title}' session={session_id}")

    def add_document(self, content: str, filename: str = "", source: str = ""):
        """Add a document chunk."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO documents (filename, content, source, created_at) VALUES (?, ?, ?, ?)",
            (filename, content, source, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info(f"Added document: {filename}")

    # ── Retrieval ──────────────────────────────────────────────

    @staticmethod
    def _fts_escape(query: str) -> str:
        """Escape a query string for FTS5 MATCH (remove special chars, join words with OR)."""
        words = re.findall(r'\w+', query)
        return " OR ".join(words) if words else ""

    def search_meetings(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search across meeting transcripts."""
        fts_query = self._fts_escape(query)
        if not fts_query:
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT m.role, m.content, m.timestamp, m.session_id, m.meeting_title, rank
            FROM meeting_turns_fts f
            JOIN meeting_turns m ON m.id = f.rowid
            WHERE meeting_turns_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def search_documents(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search across documents."""
        fts_query = self._fts_escape(query)
        if not fts_query:
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT d.id, d.filename, d.content, d.source, rank
            FROM docs_search f
            JOIN documents d ON d.id = f.rowid
            WHERE docs_search MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_recent_meetings(self, limit: int = 20) -> list[dict]:
        """Get recent meeting turns for general context."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, timestamp, meeting_title FROM meeting_turns ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    # ── RAG Pipeline ───────────────────────────────────────────

    def _build_context(self, query: str) -> str:
        """Build context string from knowledge base search results."""
        parts = []

        # Search meeting transcripts
        meeting_results = self.search_meetings(query, limit=5)
        if meeting_results:
            parts.append("=== Relevant Meeting Discussions ===")
            for r in meeting_results:
                role = r["role"].capitalize()
                title = f" [{r['meeting_title']}]" if r.get("meeting_title") else ""
                parts.append(f"{role}{title}: {r['content']}")

        # Search documents
        doc_results = self.search_documents(query, limit=3)
        if doc_results:
            parts.append("\n=== Relevant Documents ===")
            for r in doc_results:
                parts.append(f"[{r['filename']}] {r['content'][:500]}")

        # Always include recent meeting context (last 10 turns)
        recent = self.get_recent_meetings(limit=10)
        if recent:
            parts.append("\n=== Recent Meeting Context ===")
            for r in recent:
                parts.append(f"{r['role'].capitalize()}: {r['content']}")

        return "\n".join(parts) if parts else "No relevant context found in knowledge base."

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 thinking/reasoning process from output, keep only the final answer.

        Qwen3 outputs a "Thinking Process:" section followed by numbered analysis steps,
        then gives a final concise answer. We extract just the final answer.
        """
        # If text starts with "Thinking Process:", find the actual answer after the analysis
        if "Thinking Process:" in text or text.lstrip().startswith("1."):
            # Strategy: find the shortest complete sentence that directly answers the question.
            # The final answer is typically at the end, after all the numbered analysis.
            # Look for it in the last ~3 lines.
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            # Scan backwards for a short, direct answer line
            for line in reversed(lines):
                # Skip numbered analysis steps
                if re.match(r'^\d+\.', line):
                    break
                # Skip bullet points from analysis
                if line.startswith('*') and ('Task:' in line or 'Role:' in line or
                    'Constraint' in line or 'Question:' in line or 'Analyze' in line or
                    'Segment' in line or 'Extract' in line or 'Formulate' in line):
                    continue
                # Skip lines that are just analysis markers
                if line in ('Thinking Process:', '---'):
                    continue
                # Skip lines with quoted context snippets
                if line.startswith('*') and 'Context snippet' in line:
                    continue
                # Skip "**Final Check**" lines
                if '**Final Check**' in line or '**Review Constraints**' in line:
                    break
                # Skip "Final Output Construction:" pattern
                if '**Final Output Construction**' in line:
                    continue
                # This looks like the actual answer
                if len(line) > 10 and not line.startswith('*'):
                    return line

            # Fallback: look for "Final Output Construction" section
            m = re.search(r'\*\*Final Output Construction\*\*:.*?\n\s*"(.+?)"', text, re.DOTALL)
            if m:
                return m.group(1)

            # Last resort: return original
            return text

        return text

    def _call_llm(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Call local LLM via OpenAI-compatible API."""
        try:
            resp = httpx.post(
                f"{self.llm_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.llm_api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return self._strip_thinking(content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM error: {e}]"

    def query(self, user_text: str, max_tokens: int = 300) -> str:
        """
        Main RAG query: user text from STT → retrieve context → LLM generate → answer for TTS.
        """
        logger.info(f"RAG query: '{user_text}'")

        # 1. Retrieve relevant context
        context = self._build_context(user_text)

        # 2. Build prompt
        system_prompt = (
            "You are a meeting assistant. Answer the user's question based on the provided meeting context. "
            "Be concise and direct. If the context doesn't contain the answer, say so honestly. "
            "Respond in the same language the user uses."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_text}"},
        ]

        # 3. Generate answer
        answer = self._call_llm(messages, max_tokens)
        logger.info(f"RAG answer: '{answer[:100]}...'")

        # 4. Store this interaction
        self.add_meeting_turn(role="user", content=user_text)
        self.add_meeting_turn(role="assistant", content=answer)

        return answer

    # ── Utilities ──────────────────────────────────────────────

    def list_meetings(self) -> list[dict]:
        """List all meeting sessions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT session_id, meeting_title, COUNT(*) as turn_count,
                   MIN(timestamp) as start_time, MAX(timestamp) as end_time
            FROM meeting_turns GROUP BY session_id ORDER BY start_time DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def clear_all(self):
        """Clear all data (for testing)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM meeting_turns")
        conn.commit()
        conn.close()
        logger.info("Cleared all meeting turns")

    def stats(self) -> dict:
        """Get knowledge base stats."""
        conn = sqlite3.connect(self.db_path)
        meeting_count = conn.execute("SELECT COUNT(*) FROM meeting_turns").fetchone()[0]
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return {"meeting_turns": meeting_count, "documents": doc_count}


# ── Convenience: standalone test ───────────────────────────────

if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"Knowledge base stats: {kb.stats()}")
    print(f"Available models: checking...")

    try:
        resp = httpx.get(
            f"{DEFAULT_LLM_URL}/models",
            headers={"Authorization": f"Bearer {DEFAULT_LLM_KEY}"},
            timeout=5.0,
        )
        models = [m["id"] for m in resp.json().get("data", [])]
        print(f"LLM models: {models}")
    except Exception as e:
        print(f"LLM connection failed: {e}")
