#!/usr/bin/env python3
"""
Zoom Captions Database Query 服务器 + HTTP API LLM
从 captions.db 查询历史字幕，结合本地 Qwen3.5-2B-MLX-8bit 模型回答问题

问答流程：
1. 使用 LLM 从问题中提取关键词
2. 根据关键词搜索相关字幕
3. 构建上下文（会议信息、发言人、时间）
4. 使用 LLM 基于上下文生成友好回答

用法: python3 zoom_captions_server.py

数据源:
- /Volumes/sn7100/jerry/code/zoom-meeting-sdk-demo/captions.db
- LLM API: http://127.0.0.1:12345/v1 (Qwen3.5-2B-MLX-8bit)
"""

import asyncio
import websockets
import json
import logging
import sqlite3
import os
import re
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("zoom-captions-server")

# --- Zoom Captions Database ---
ZOOM_DB_PATH = "/Volumes/sn7100/jerry/code/zoom-meeting-sdk-demo/captions.db"


class CaptionsDB:
    """Zoom 字幕数据库访问层"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.connect()

    def connect(self):
        """连接数据库"""
        if not os.path.exists(self.db_path):
            log.warning(f"Database not found: {self.db_path}")
            self.conn = None
            return

        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            log.info(f"Connected to captions DB: {self.db_path}")
        except Exception as e:
            log.error(f"Failed to connect to database: {e}")
            self.conn = None

    def is_connected(self) -> bool:
        return self.conn is not None

    def search_captions(self, keyword: str, limit: int = 20) -> List[Dict]:
        """搜索包含关键词的字幕"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT speaker, text, received_at
                FROM captions
                WHERE text LIKE ?
                ORDER BY received_at DESC
                LIMIT ?
            """, (f"%{keyword}%", limit))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Error searching captions: {e}")
            return []

    def execute_sql(self, sql: str, params: tuple = ()) -> List[Dict]:
        """执行自定义 SQL 查询"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Error executing SQL: {e}")
            return []

    def get_schema(self) -> str:
        """获取数据库结构信息"""
        return """
数据库结构:
- meetings 表: id (主键), meeting_number, topic, host_name, started_at, ended_at
- captions 表: id (主键), meeting_id (外键), speaker, text, caption_type, received_at
- participants 表: id (主键), meeting_id (外键), user_id, user_name, is_host, joined_at

使用说明:
1. speaker 字段包含发言者名称
2. text 字段包含发言内容
3. received_at 字段包含时间戳
4. meeting_id 关联到 meetings 表
"""

    def get_recent_captions(self, minutes: int = 30, limit: int = 50) -> List[Dict]:
        """获取最近 N 分钟的字幕"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cutoff_time = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
                SELECT speaker, text, received_at
                FROM captions
                WHERE received_at >= ?
                ORDER BY received_at DESC
                LIMIT ?
            """, (cutoff_time, limit))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Error getting recent captions: {e}")
            return []

    def get_meeting_captions(self, meeting_id: int, limit: int = 100) -> List[Dict]:
        """获取指定会议的字幕"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT speaker, text, received_at
                FROM captions
                WHERE meeting_id = ?
                ORDER BY received_at ASC
                LIMIT ?
            """, (meeting_id, limit))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Error getting meeting captions: {e}")
            return []

    def get_all_meetings(self) -> List[Dict]:
        """获取所有会议列表"""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, meeting_number, topic, host_name, started_at
                FROM meetings
                ORDER BY started_at DESC
            """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Error getting meetings: {e}")
            return []

    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        if not self.conn:
            return {}

        try:
            cursor = self.conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM meetings")
            meeting_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM captions")
            caption_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM participants")
            participant_count = cursor.fetchone()["count"]

            return {
                "meetings": meeting_count,
                "captions": caption_count,
                "participants": participant_count
            }
        except Exception as e:
            log.error(f"Error getting stats: {e}")
            return {}


# --- 初始化数据库连接 ---
captions_db = CaptionsDB(ZOOM_DB_PATH)
CAPTIONS_OK = captions_db.is_connected()

if CAPTIONS_OK:
    stats = captions_db.get_stats()
    log.info(f"Captions DB: {stats.get('meetings', 0)} meetings, {stats.get('captions', 0)} captions")
else:
    log.warning("Captions database not available")

# --- 线程池执行器 (用于 LLM 推理) ---
executor = ThreadPoolExecutor(max_workers=1)

# --- 内存字幕缓存 (实时字幕，避免数据库延迟) ---
class CaptionCache:
    """实时字幕缓存，避免数据库写入延迟"""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.captions: List[Dict] = []
        self.lock = None

    async def init(self):
        """初始化异步锁"""
        import asyncio
        self.lock = asyncio.Lock()

    async def add(self, speaker: str, text: str, timestamp: str = None):
        """添加字幕到缓存"""
        if not self.lock:
            await self.init()

        async with self.lock:
            from datetime import datetime
            caption = {
                "speaker": speaker,
                "text": text,
                "received_at": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.captions.append(caption)

            # 保持缓存大小
            if len(self.captions) > self.max_size:
                self.captions = self.captions[-self.max_size:]

            # 打印日志，方便测试
            log.info(f"[Cache] Added caption [{len(self.captions)}/{self.max_size}]: {speaker}: {text[:50]}...")

    async def search(self, keywords: List[str], limit: int = 50) -> List[Dict]:
        """根据关键词搜索缓存中的字幕"""
        if not self.lock:
            await self.init()

        async with self.lock:
            if not keywords:
                return self.captions[-limit:]

            # 搜索包含任一关键词的字幕
            results = []
            for cap in self.captions:
                text_lower = cap["text"].lower()
                if any(kw.lower() in text_lower for kw in keywords):
                    results.append(cap)

            return results[-limit:]

    def get_recent(self, limit: int = 50) -> List[Dict]:
        """获取最近 N 条字幕（同步方法，用于 LLM 上下文构建）"""
        return self.captions[-limit:]


# 全局缓存实例
caption_cache = CaptionCache(max_size=200)


def format_captions_response(captions: List[Dict], query: str = "") -> str:
    """Format caption query results as natural language response (fallback when LLM unavailable)"""
    if not captions:
        return f"Sorry, I couldn't find any relevant information about '{query}' in the meeting records."

    # Extract information from captions and build a knowledge-style response
    if len(captions) == 1:
        c = captions[0]
        text = c.get('text', '')
        # Use content directly as answer, avoid 'X said' format
        return text.strip()
    else:
        # Multiple results, combine content
        texts = [c.get('text', '') for c in captions[:3]]
        # Combine relevant content
        combined = "; ".join(texts)
        return combined.strip()


# --- 唤醒词列表 ---
TRIGGER_WORDS = ["hey echo", "echo", "Aiko", "mike", "mac", "mic"]

# --- 本地 LLM 模型配置 (HTTP API) ---
LLM_API_BASE = "http://127.0.0.1:12345/v1"
LLM_API_KEY = "1234"
LLM_MODEL = "Qwen3.5-2B-MLX-8bit"
MAX_TOKENS = 500
TEMPERATURE = 0.7

# --- 本地 LLM (HTTP API) ---
LLM_OK = False


def check_llm_connection():
    """检查 LLM API 连接"""
    global LLM_OK
    try:
        response = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5
            },
            timeout=5
        )
        LLM_OK = response.status_code == 200
        return LLM_OK
    except Exception as e:
        LLM_OK = False
        return False


def call_llm(messages: list) -> Optional[str]:
    """使用 HTTP API 调用 LLM"""
    if not LLM_OK:
        return None

    try:
        response = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"LLM API 错误: {e}")
        return None


def call_llm_with_context(query: str, context: str) -> Optional[str]:
    """Use LLM to answer questions based on context"""
    if not context:
        context = "No relevant captions found."

    system_prompt = """You are a professional meeting knowledge assistant. Extract information from meeting conversations and provide knowledge-based answers.

Core Principles:
1. **Extract knowledge, don't paraphrase dialogue** - Don't say 'X said...', just state the facts directly
2. **Synthesize scattered information** - If information is scattered across multiple places, combine into a complete answer
3. **Answer directly** - If asked about price, give the price; if asked about time, give the time
4. **Natural conversation** - Answer in your own words, as if you already know this information
5. **Reference only when necessary** - Only mention 'according to the meeting' when the answer is uncertain

Answer Format:
- Give the answer directly, no prefixes like 'According to captions...' or 'Meeting records show...'
- For specific info like prices, times, numbers, provide them directly
- If information is insufficient, clearly state what's missing, don't guess
- Keep it concise, typically 1-3 sentences

Examples:
- Q: How much is the book? A: The book costs XX yuan.
- Q: When is the meeting? A: The meeting is scheduled for tomorrow at 2 PM.
- Q: Who is responsible for this project? A: Zhang San is responsible for this project.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Meeting conversation records:\n{context}\n\nQuestion: {query}"}
    ]

    return call_llm(messages)


def extract_keywords(question: str) -> List[str]:
    """让 LLM 提取问题中的关键词"""
    if not LLM_OK:
        # 简单分词作为回退
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', question)
        return [w for w in words if len(w) > 1]

    messages = [
        {
            "role": "system",
            "content": "You are a keyword extraction assistant. Extract the 2-4 most important Chinese or English keywords from the user's question, separated by spaces. Return only the keywords, nothing else."
        },
        {
            "role": "user",
            "content": f"问题: {question}\n\n关键词:"
        }
    ]
    response = call_llm(messages)
    if response:
        keywords = response.strip().split()
        return [kw for kw in keywords if len(kw) > 1]
    return []


async def search_captions_by_keywords(keywords: List[str]) -> tuple[List[Dict], bool]:
    """根据关键词搜索相关字幕

    优先从内存缓存获取，缓存为空时才查询数据库。

    Returns:
        (字幕列表, 是否来自缓存)
    """
    # 先从内存缓存获取（实时字幕）
    cached = await caption_cache.search(keywords, limit=50)
    if cached:
        log.info(f"[Cache] Found {len(cached)} captions from memory cache")
        return cached, True

    # 缓存为空，回退到数据库查询（历史字幕）
    if not captions_db.is_connected():
        return [], False

    if not keywords:
        # 获取最近的字幕
        return captions_db.get_recent_captions(minutes=1440, limit=50), False  # 24小时内

    import sqlite3
    conn = sqlite3.connect(ZOOM_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 构建模糊匹配条件
    conditions = []
    params = []
    for kw in keywords:
        conditions.append("c.text LIKE ?")
        params.append(f"%{kw}%")

    query = f"""
        SELECT c.text, c.speaker, c.received_at, m.topic, m.meeting_number
        FROM captions c
        JOIN meetings m ON c.meeting_id = m.id
        WHERE {' OR '.join(conditions)}
        ORDER BY c.received_at
        LIMIT 50
    """

    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results, False


def build_context_from_captions(captions: List[Dict]) -> str:
    """从字幕列表构建上下文"""
    if not captions:
        return "暂无会议记录。"

    context_parts = []
    current_meeting = None

    for cap in captions:
        meeting_info = f"会议: {cap.get('topic') or cap.get('meeting_number', 'N/A')}"
        if meeting_info != current_meeting:
            current_meeting = meeting_info
            context_parts.append(f"\n[{meeting_info}]")

        speaker = cap.get('speaker', '未知')
        time = cap.get('received_at', '')[-8:]  # 只显示时间部分
        text = cap.get('text', '')
        context_parts.append(f"  [{time}] {speaker}: {text}")

    return "\n".join(context_parts)


def is_general_question(question: str) -> bool:
    """判断是否为普通问题（不需要查询数据库）

    普通问题：问候、闲聊、一般性问题
    具体问题：会议内容、数据、事实查询
    """
    if not LLM_OK:
        return False

    # 检测是否为具体问题的关键词
    specific_keywords = [
        # 会议相关
        "meeting", "meeting content", "what discussed", "what said", "who said",
        "会议", "会议内容", "讨论了", "说了什么", "谁说",
        # 数据相关
        "price", "cost", "how much", "number", "count", "when", "where", "who",
        "价格", "多少钱", "多少", "时间", "地点", "谁",
        # 记录相关
        "record", "history", "captions", "mentioned", "talked about",
        "记录", "历史", "字幕", "提到", "讨论"
    ]

    question_lower = question.lower()
    for kw in specific_keywords:
        if kw in question_lower:
            return False  # 是具体问题，需要查询数据库

    # 检测是否为问候/闲聊
    general_patterns = [
        "hello", "hi", "hey", "how are you", "what's up", "thank", "thanks",
        "你好", "嗨", "谢谢", "怎么样",
    ]

    for pattern in general_patterns:
        if pattern in question_lower:
            return True  # 是普通问题

    # 默认情况下，如果问题很短且不包含具体关键词，视为普通问题
    return len(question.split()) <= 3


async def answer_question(question: str) -> tuple[Optional[str], Optional[List[Dict]]]:
    """基于关键词搜索回答问题（优先使用内存缓存）"""
    # 判断问题类型
    if is_general_question(question):
        log.info(f"[General Question] 直接由 LLM 回答: {question}")

        messages = [
            {
                "role": "system",
                "content": """You are a helpful meeting assistant. Answer the user's question directly and naturally.
Keep your response concise (1-2 sentences). Be friendly and helpful."""
            },
            {"role": "user", "content": question}
        ]

        answer = call_llm(messages)
        if answer:
            return answer, None
        # 如果 LLM 失败，回退到数据库查询
        log.info("LLM direct answer failed, falling back to database query")

    # 具体问题：查询数据库
    keywords = extract_keywords(question)
    log.info(f"提取关键词: {keywords}")

    # 搜索相关内容（优先从内存缓存）
    relevant_captions, from_cache = await search_captions_by_keywords(keywords)
    source = "[Cache]" if from_cache else "[Database]"
    log.info(f"{source} Found {len(relevant_captions)} relevant captions")

    if not relevant_captions:
        return "Sorry, I couldn't find any relevant content in the meeting records.", None

    # 构建上下文
    context = build_context_from_captions(relevant_captions)

    # 构建提示词
    system_prompt = """You are a professional meeting knowledge assistant. Extract information from meeting conversations and provide knowledge-based answers.

Core Principles:
1. **Extract knowledge, don't paraphrase dialogue** - Don't say 'X said...', just state the facts directly
2. **Synthesize scattered information** - If information is scattered across multiple places, combine into a complete answer
3. **Answer directly** - If asked about price, give the price; if asked about time, give the time
4. **Natural conversation** - Answer in your own words, as if you already know this information
5. **Reference only when necessary** - Only mention 'according to the meeting' when the answer is uncertain

Answer Format:
- Give the answer directly, no prefixes like 'According to captions...' or 'Meeting records show...'
- For specific info like prices, times, numbers, provide them directly
- If information is insufficient, clearly state what's missing, don't guess
- Keep it concise, typically 1-3 sentences

Examples:
- Q: How much is the book? A: The book costs XX yuan.
- Q: When is the meeting? A: The meeting is scheduled for tomorrow at 2 PM.
- Q: Who is responsible for this project? A: Zhang San is responsible for this project.

Meeting conversation records:
""" + context

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    answer = call_llm(messages)
    if not answer:
        answer = format_captions_response(relevant_captions, question)

    return answer, relevant_captions


# 初始化 LLM 连接
check_llm_connection()
if LLM_OK:
    log.info(f"✓ LLM API 连接成功: {LLM_API_BASE}")
else:
    log.warning(f"✗ LLM API 不可用: {LLM_API_BASE}")

# --- WebSocket ---
# 维护所有连接的客户端，用于广播
connected_clients = []


async def broadcast_message(message: dict, exclude_ws=None):
    """广播消息给所有连接的客户端"""
    if not connected_clients:
        log.warning("[Broadcast] 没有连接的客户端")
        return

    data = json.dumps(message)
    disconnected = []

    for client in connected_clients:
        if client == exclude_ws:
            continue
        try:
            await client.send(data)
            log.info(f"[Broadcast] 发送消息到客户端")
        except Exception as e:
            log.error(f"[Broadcast] 发送失败: {e}")
            disconnected.append(client)

    # 清理断开的连接
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)


async def handler(ws):
    log.info("客户端连接")

    # 添加到连接列表
    connected_clients.append(ws)
    remote_addr = ws.remote_address if hasattr(ws, 'remote_address') else 'unknown'
    log.info(f"[WebSocket] 新连接来自: {remote_addr}, 总连接数: {len(connected_clients)}")

    # 发送初始化状态
    await ws.send(json.dumps({
        "type": "ready",
        "captions": CAPTIONS_OK,
        "llm": LLM_OK,
        "stats": captions_db.get_stats() if CAPTIONS_OK else {}
    }))

    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            # 文本查询 (检测唤醒词)
            if msg_type == "text":
                text = data.get("text", "").strip()
                if not text:
                    continue

                log.info(f"收到文本: {text}")

                # 检测唤醒词
                text_lower = text.lower()
                trigger_word = None
                question = text

                for word in TRIGGER_WORDS:
                    if word in text_lower:
                        trigger_word = word
                        question = text_lower.split(word, 1)[1].strip()
                        break

                # 只有检测到唤醒词才回答
                if trigger_word:
                    log.info(f"唤醒词触发: {trigger_word}, 问题: {question}")

                    answer, captions = await answer_question(question)

                    await ws.send(json.dumps({
                        "type": "answer",
                        "text": answer,
                        "question": question,
                        "captions": (captions or [])[:5]
                    }))
                else:
                    log.info(f"没有唤醒词，忽略: {text}")
                    await ws.send(json.dumps({
                        "type": "no_trigger",
                        "text": f"请使用唤醒词: {', '.join(TRIGGER_WORDS)}"
                    }))

            # 获取会议列表
            elif msg_type == "meetings":
                meetings = captions_db.get_all_meetings()
                await ws.send(json.dumps({
                    "type": "meetings",
                    "data": meetings
                }))

            # 获取指定会议的字幕
            elif msg_type == "meeting_captions":
                meeting_id = data.get("meeting_id")
                if meeting_id:
                    captions = captions_db.get_meeting_captions(meeting_id)
                    await ws.send(json.dumps({
                        "type": "meeting_captions",
                        "meeting_id": meeting_id,
                        "data": captions
                    }))

            # 获取最近字幕
            elif msg_type == "recent_captions":
                minutes = data.get("minutes", 30)
                captions = captions_db.get_recent_captions(minutes=minutes)
                await ws.send(json.dumps({
                    "type": "recent_captions",
                    "data": captions
                }))

            # Zoom 实时字幕 - 检测唤醒词
            elif msg_type == "caption":
                text = data.get("text", "").strip()
                speaker = data.get("speaker", "")
                if not text:
                    continue

                log.info(f"收到字幕: {speaker}: {text}")

                # 立即添加到内存缓存
                await caption_cache.add(speaker, text)

                # 检测唤醒词
                text_lower = text.lower()
                trigger_word = None
                question = None

                for word in TRIGGER_WORDS:
                    if word in text_lower:
                        trigger_word = word
                        question = text_lower.split(word, 1)[1].strip()
                        break

                # 如果检测到唤醒词且有后续内容
                if trigger_word and question:
                    log.info(f"唤醒词触发: {trigger_word}, 问题: {question}")

                    # 调用 LLM 查询
                    answer, captions = await answer_question(question)

                    # 广播答案给所有客户端（包括 voice-asr-parakeet.html）
                    await broadcast_message({
                        "type": "answer",
                        "text": answer,
                        "question": question,
                        "captions": (captions or [])[:5]
                    })
                    log.info(f"[Broadcast] 答案已广播给 {len(connected_clients)} 个客户端")
                else:
                    # 仅确认收到字幕
                    await ws.send(json.dumps({
                        "type": "caption_received",
                        "text": text
                    }))

            # Ping/Pong
            elif msg_type == "ping":
                await ws.send(json.dumps({"type": "pong"}))

    finally:
        # 客户端断开，清理连接
        if ws in connected_clients:
            connected_clients.remove(ws)
            log.info(f"[WebSocket] 客户端断开，剩余连接: {len(connected_clients)}")


async def main():
    log.info("=" * 50)
    log.info("Zoom Captions Database Query + 本地 LLM 服务器")
    log.info("  ws://localhost:8767")
    log.info(f"  Captions DB: {'就绪' if CAPTIONS_OK else '不可用'}")
    log.info(f"  本地 LLM: {'就绪' if LLM_OK else '不可用'}")

    if CAPTIONS_OK:
        stats = captions_db.get_stats()
        log.info(f"  数据库: {stats.get('meetings', 0)} 会议, {stats.get('captions', 0)} 字幕")

    log.info("=" * 50)

    async with websockets.serve(handler, "localhost", 8767):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
