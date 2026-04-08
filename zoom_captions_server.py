#!/usr/bin/env python3
"""
Zoom Captions Database Query 服务器 + 本地 LLM
从 captions.db 查询历史字幕，结合本地 Qwen3.5-2B-MLX-8bit 模型回答问题

用法: uv run zoom_captions_server.py

数据源:
- /Volumes/sn7100/jerry/code/zoom-meeting-sdk-demo/captions.db
- 本地模型: Qwen3.5-2B-MLX-8bit
"""

import asyncio
import websockets
import json
import logging
import sqlite3
import os
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("zoom-captions-server")

# --- 本地 LLM 模型配置 ---
# 使用本地模型目录
MODEL_PATH = "/Volumes/sn7100/jerry/code/mlx-vlm"
MAX_TOKENS = 150
TEMPERATURE = 0.7

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


def format_captions_response(captions: List[Dict], query: str = "") -> str:
    """格式化字幕查询结果为自然语言回答（备用，不使用 LLM）"""
    if not captions:
        return f"Sorry, I couldn't find any information about '{query}' in the meeting captions."

    # 构建回答
    if len(captions) == 1:
        c = captions[0]
        return f"Found: {c.get('speaker', 'Speaker')} said \"{c.get('text', '')}\""
    else:
        # 多条结果，汇总
        speakers = set(c.get('speaker', 'Speaker') for c in captions)
        texts = [c.get('text', '') for c in captions[:5]]

        if len(speakers) == 1:
            speaker = list(speakers)[0]
            return f"{speaker} mentioned: {', '.join(texts[:3])}"
        else:
            return f"Found {len(captions)} related captions. Recent mentions: {', '.join(texts[:3])}"


# --- 本地 LLM (MLX) ---
model = None
tokenizer = None
LLM_OK = False

try:
    from mlx_lm import load, generate

    def init_llm():
        global model, tokenizer, LLM_OK
        try:
            log.info(f"正在加载 MLX 模型: {MODEL_PATH}")
            log.info("首次加载可能需要下载模型，请耐心等待...")
            model, tokenizer = load(MODEL_PATH)
            LLM_OK = True
            log.info("✓ MLX 模型加载成功")
        except Exception as e:
            log.error(f"✗ MLX 模型加载失败: {e}")
            log.info("提示: 安装 mlx-lm: pip install mlx-lm")
            LLM_OK = False

    def call_llm(query: str, context: str) -> Optional[str]:
        """使用本地 MLX 模型生成回答"""
        if not LLM_OK or not model or not tokenizer:
            return None

        if not context:
            context = "No relevant captions found."

        system_prompt = "You are a helpful assistant that answers questions based on meeting captions. Keep answers concise (2-3 sentences)."

        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Context from meeting captions:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""

        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                verbose=False
            )
            # 移除 prompt 部分，只返回生成的回答
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            return response.strip()
        except Exception as e:
            log.error(f"LLM 生成错误: {e}")
            return None

    def call_llm_with_sql(query: str) -> tuple[Optional[str], Optional[str], Optional[List[Dict]]]:
        """使用 LLM 生成 SQL 查询，执行并回答"""
        if not LLM_OK or not model or not tokenizer:
            return None, None, None

        schema = captions_db.get_schema()

        system_prompt = """You are a SQL expert. Based on the user's question, generate a SQLite query to search the meeting captions database.

Rules:
1. Return ONLY the SQL query, no explanations, no thinking process
2. Do NOT use <think> tags or any reasoning tags
3. Use SELECT with speaker, text, received_at columns
4. Use LIKE for text search with % wildcards
5. Limit results to 20 rows maximum
6. Order by received_at DESC (most recent first)
7. Return valid SQL only"""

        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{schema}

Question: {query}

Generate SQL query:<|im_end|>
<|im_start|>assistant
"""

        try:
            # 第一步：生成 SQL
            sql_response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=100,
                verbose=False
            )

            # 清理 SQL 响应 - 移除所有多余内容
            # 1. 移除 <|im_start|>assistant 标记
            if "<|im_start|>assistant" in sql_response:
                sql_response = sql_response.split("<|im_start|>assistant")[-1].strip()

            # 2. 移除 <|im_end|> 标记
            if "<|im_end|>" in sql_response:
                sql_response = sql_response.split("<|im_end|>")[0].strip()

            # 3. 移除  标签及其内容（Qwen 的思考过程）
            while "" in sql_response:
                before = sql_response
                # 移除从  到  的内容
                sql_response = re.sub(r'<think>.*?</think>\s*', '', sql_response, flags=re.DOTALL)
                if sql_response == before:
                    break  # 没有更多变化，退出循环

            # 4. 移除 markdown 代码块标记 ```sql 或 ```
            # 移除 ```sql 或 ```SQL 开头
            sql_response = re.sub(r'^```(?:sql|SQL)?\s*', '', sql_response, flags=re.IGNORECASE)
            # 移除结尾的 ```
            sql_response = re.sub(r'```\s*$', '', sql_response)

            # 5. 清理重复内容 - 只保留到第一个分号
            if ';' in sql_response:
                sql_response = sql_response.split(';')[0] + ';'

            # 6. 移除重复的 LIMIT 子句
            # 查找所有 LIMIT 并只保留第一个
            limit_matches = list(re.finditer(r'LIMIT\s+\d+', sql_response, re.IGNORECASE))
            if len(limit_matches) > 1:
                # 只保留第一个 LIMIT，移除其他的
                first_limit_end = limit_matches[0].end()
                sql_response = sql_response[:first_limit_end] + ';'

            # 7. 最终清理空行和多余空格
            lines = [line.strip() for line in sql_response.split('\n') if line.strip()]
            sql_response = '\n'.join(lines).strip()

            log.info(f"生成的 SQL: {sql_response}")

            # 第二步：执行 SQL
            captions = captions_db.execute_sql(sql_response)

            if not captions:
                return f"Sorry, couldn't find relevant information.", sql_response, None

            # 第三步：基于结果生成回答
            context = build_context_from_captions(captions)
            answer = call_llm(query, context)

            if answer:
                return answer, sql_response, captions
            else:
                # 回退到简单格式
                fallback = format_captions_response(captions, query)
                return fallback, sql_response, captions

        except Exception as e:
            log.error(f"LLM+SQL 错误: {e}")
            return None, None, None

    # 初始化模型
    init_llm()

except ImportError as e:
    log.warning(f"✗ mlx-lm 未安装: {e}")
    log.info("提示: 安装 mlx-lm: pip install mlx-lm")
    LLM_OK = False

    def call_llm(query: str, context: str) -> Optional[str]:
        return None

    def init_llm():
        pass


def build_context_from_captions(captions: List[Dict]) -> str:
    """从字幕列表构建上下文"""
    if not captions:
        return "No relevant captions found."

    context_parts = []
    for c in captions[:10]:  # 最多使用 10 条字幕
        speaker = c.get('speaker', 'Speaker')
        text = c.get('text', '')
        time_str = c.get('received_at', '')
        context_parts.append(f"[{speaker}]: {text}")

    return "\n".join(context_parts)


# --- WebSocket ---
async def handler(ws):
    log.info("客户端连接")

    # 发送初始化状态
    await ws.send(json.dumps({
        "type": "ready",
        "captions": CAPTIONS_OK,
        "llm": LLM_OK,
        "stats": captions_db.get_stats() if CAPTIONS_OK else {}
    }))

    async for msg in ws:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type")

        # 文本查询
        if msg_type == "text":
            text = data.get("text", "").strip()
            if not text:
                continue

            log.info(f"查询: {text}")

            # 使用 LLM 生成 SQL 并回答
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, call_llm_with_sql, text
            )

            if result:
                answer, sql, captions = result

                if answer:
                    await ws.send(json.dumps({
                        "type": "answer",
                        "text": answer,
                        "sql": sql,
                        "captions": (captions or [])[:5]
                    }))
                else:
                    # 回退到简单搜索
                    captions = captions_db.search_captions(text, limit=10)
                    if captions:
                        fallback = format_captions_response(captions, text)
                        await ws.send(json.dumps({
                            "type": "answer",
                            "text": fallback,
                            "captions": captions[:5]
                        }))
                    else:
                        await ws.send(json.dumps({
                            "type": "answer",
                            "text": f"Sorry, I couldn't find any information about '{text}' in the meeting captions."
                        }))
            else:
                # LLM 不可用，回退到简单搜索
                captions = captions_db.search_captions(text, limit=10)
                if captions:
                    fallback = format_captions_response(captions, text)
                    await ws.send(json.dumps({
                        "type": "answer",
                        "text": fallback,
                        "captions": captions[:5]
                    }))
                else:
                    await ws.send(json.dumps({
                        "type": "answer",
                        "text": f"Sorry, I couldn't find any information about '{text}' in the meeting captions."
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

        # Ping/Pong
        elif msg_type == "ping":
            await ws.send(json.dumps({"type": "pong"}))


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
