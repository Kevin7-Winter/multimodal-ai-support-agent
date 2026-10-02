from __future__ import annotations

import json
import base64
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()


app = FastAPI(
    title="Anker Power Bank Support API",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "anker_support.db"
WORK_ORDER_PATH = BASE_DIR / "work_orders.json"
KB_VECTOR_PATH = DATA_DIR / "kb_vectors.json"
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base" / "powerbank_kb.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE = 8 * 1024 * 1024

CASES: Dict[str, "CaseState"] = {}

if FRONTEND_DIR.exists():
    app.mount(
        "/app",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads",
)


class CaseState(BaseModel):
    case_id: str

    status: Literal[
        "collecting",
        "diagnosing",
        "resolved",
        "handoff",
    ] = "collecting"

    product_category: str = "充电宝"
    product_model: Optional[str] = None

    intent: Optional[str] = None
    emotion: Optional[str] = None
    emotion_level: Literal[
        "low",
        "medium",
        "high",
    ] = "low"

    symptoms: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    forbidden_actions: List[str] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)

    next_action: str = "请描述充电宝型号和具体现象"
    resolution: Optional[str] = None
    handoff_reason: Optional[str] = None

    official_sources: List[dict] = Field(default_factory=list)


    historical_cases: List[dict] = Field(default_factory=list)
    has_similar_cases: bool = False
    tool_results: List[dict] = Field(default_factory=list)

    image_evidence: List[dict] = Field(default_factory=list)
    photos_requested: bool = False
    photos_received: bool = False

    current_node: str = "start"

    transcript: List[dict] = Field(default_factory=list)
    updated_at: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    case_id: Optional[str] = None


class ChatResponse(BaseModel):
    case: CaseState
    reply: str
    action: str
    risk_level: Literal["none", "high"]
    citations: List[str] = Field(default_factory=list)


class HandoffResponse(BaseModel):
    case_id: str
    summary: str
    status: Literal["handoff"]


class ImageAnalysisRequest(BaseModel):
    model_name: Optional[str] = None
    serial_number: Optional[str] = None
    indicator_status: Optional[str] = None
    visible_risk: Optional[str] = None
    description: str = ""


class OrderLookupRequest(BaseModel):
    order_id: str = Field(min_length=3, max_length=80)


class WarrantyLookupRequest(BaseModel):
    product_model: str = Field(min_length=2, max_length=40)
    purchase_date: Optional[str] = None
    order_id: Optional[str] = None
    serial_number: Optional[str] = None


class SerialLookupRequest(BaseModel):
    serial_number: str = Field(min_length=4, max_length=80)


class DealerOrderLookupRequest(BaseModel):
    order_id: str = Field(min_length=3, max_length=80)
    dealer: Optional[str] = None


class SupportGraphState(TypedDict, total=False):
    case_id: str
    message: str

    reply: str
    action: str
    risk_level: Literal["none", "high"]
    citations: List[str]

    route: str
    current_node: str


RISK_RE = re.compile(
    r"鼓包|膨胀|冒烟|起火|异味|刺鼻|漏液|进水|"
    r"异常发热|很烫|过热|变形|烧焦"
)

SAFETY_ACTIONS = [
    "立即停止使用和充电",
    "不要拆解、挤压或刺穿设备",
    "将设备放在远离人员和可燃物的安全位置",
    "联系人工售后处理",
]

RISK_TERM_PATTERN = (
    r"(?:鼓包|膨胀|变形|冒烟|起火|异味|刺鼻|焦糊|烧焦|"
    r"漏液|液体泄漏|进水|受潮|掉水|异常发热|很烫|烫手|"
    r"过热|发烫|发热)"
)

NEGATED_RISK_RE = re.compile(
    rf"(?:没有|没|无|未|不|并未|并无|未见|未发现|"
    rf"没有发现|不会|不可能|也没有|肯定没有|应该没有)"
    rf"[^，。；,.!?！？]{{0,8}}{RISK_TERM_PATTERN}"
)

POST_NEGATED_RISK_RE = re.compile(
    rf"{RISK_TERM_PATTERN}"
    rf"[^，。；,.!?！？]{{0,6}}"
    rf"(?:没有|没|无|未|不|不是|应该没有|肯定没有|可能没有)"
)


def strip_negated_risk_mentions(message: str) -> str:
    """Remove explicitly negated risk phrases before rule matching."""
    effective = NEGATED_RISK_RE.sub("", message)
    return POST_NEGATED_RISK_RE.sub("", effective)


def has_negated_risk_mention(message: str) -> bool:
    return bool(
        NEGATED_RISK_RE.search(message)
        or POST_NEGATED_RISK_RE.search(message)
    )


def safety_risk_type(message: str) -> Optional[str]:
    checks = [
        (r"起火|冒烟", "冒烟或起火"),
        (r"鼓包|膨胀|变形", "电池鼓包或变形"),
        (r"漏液|液体泄漏", "电池漏液"),
        (r"异味|刺鼻|焦糊|烧焦", "异常气味或烧焦"),
        (r"进水|受潮|掉水", "进水或受潮"),
        (r"异常发热|很烫|烫手|过热", "异常发热"),
    ]
    effective_message = strip_negated_risk_mentions(message)

    for pattern, label in checks:
        if re.search(pattern, effective_message, re.IGNORECASE):
            return label
    return None

CHARGE_RE = re.compile(
    r"充电宝|移动电源|充不进|无法充电|不充电|"
    r"充电异常|充电很慢|电量不涨|充电没反应"
)

OUTPUT_RE = re.compile(
    r"不能充手机|无法给手机充电|没有输出|不输出|"
    r"手机充不上|接口没反应|USB没反应"
)

MODEL_RE = re.compile(
    r"(?<![A-Z0-9])([ABC]\d{3,4}[A-Z]?)(?![A-Z0-9])",
    re.IGNORECASE,
)

HIGH_EMOTION_WORDS = [
    "投诉",
    "曝光",
    "报警",
    "骗人",
    "气死",
    "太垃圾",
    "我要投诉",
    "非常生气",
    "垃圾",
    "离谱",
    "黑店",
    "欺骗",
    "愤怒",
    "我靠",
    "卧槽",
    "他妈的",
    "妈的",
    "什么破",
    "破设备",
    "破东西",
    "破烂",
    "干爆",
    "炸了",
    "废物",
]

MEDIUM_EMOTION_WORDS = [
    "着急",
    "急用",
    "怎么办",
    "崩溃",
    "耽误",
    "很烦",
    "担心",
    "害怕",
    "服了",
    "生气",
    "火大",
    "失望",
    "搞什么",
    "怎么搞",
    "怎么回事",
    "什么意思",
    "无法接受",
    "质量太差",
    "太差劲",
]

HIGH_EMOTION_RE = re.compile(
    r"我操(?!作)|卧槽|我草|草泥马|操你(?:妈|全家)?|"
    r"去你妈的?|你妈的?|他妈的|妈了个|妈的|"
    r"傻逼|傻B|煞笔|脑残|智障|神经病|"
    r"狗东西|畜生|王八蛋|废物|滚蛋|滚你|"
    r"狗屁|放屁|什么玩意|破玩意|垃圾玩意|破烂玩意|"
    r"投诉|曝光|报警|起诉|维权|12315|315|消费者协会|"
    r"法院|律师|发网上|网上曝光|假货|诈骗"
)

MEDIUM_EMOTION_RE = re.compile(
    r"我服了|服了|无语|离谱|恶心|烦死|气人|气炸|"
    r"火大|崩溃|受不了|忍不了|郁闷|心累|糟心|"
    r"失望|太失望|差评|什么问题|什么质量|就这质量|"
    r"才[0-9一二三四五六七八九十]+(?:天|小时|周).{0,8}"
    r"(?:坏|裂|故障|不能|不行)|"
    r"怎么搞|搞什么|怎么回事|什么意思|无法接受|"
    r"太差|质量太差|太差劲|退钱|退款|退货|退换"
)

MEDIUM_EMOTION_CATEGORY_RULES = [
    (
        "担忧或害怕",
        re.compile(
            r"担心|担忧|害怕|恐惧|恐慌|紧张|不安|心慌|"
            r"后怕|提心吊胆|忧心|担心坏|会不会爆炸|"
            r"会不会出事|人身安全|危险|吓人|吓死|可怕"
        ),
    ),
    (
        "着急或急迫",
        re.compile(
            r"着急|急死|急用|尽快|马上|立刻|来不及|"
            r"等不了|赶时间|赶飞机|明天要用|今天要用|"
            r"催一下|催催|什么时候能|多久能|快点"
        ),
    ),
    (
        "失望或不满",
        re.compile(
            r"失望|太失望|不满意|很差|太差|差劲|敷衍|"
            r"不专业|效率低|拖太久|处理太慢|差评|无语|"
            r"离谱|坑人|被坑|糟心|恶心|问题还在|"
            r"还是不行|仍然不行|没有解决|没解决"
        ),
    ),
    (
        "无奈或崩溃",
        re.compile(
            r"崩溃|受不了|忍不了|烦死|心累|无奈|没辙|"
            r"不知道怎么办|如何是好|怎么办|头疼|郁闷|"
            r"难受|委屈|快疯了|烦人|烦死了"
        ),
    ),
    (
        "困惑或不解",
        re.compile(
            r"搞不懂|看不懂|弄不明白|不明白|不理解|"
            r"一头雾水|莫名其妙|怎么回事|怎么搞|"
            r"什么问题|什么情况|啥情况"
        ),
    ),
]

POSITIVE_EMOTION_RE = re.compile(
    r"谢谢|感谢|多谢|辛苦了|满意|挺满意|很好|不错|"
    r"解决了|修好了|恢复了|可以用了|正常了|没问题了|"
    r"明白|收到|理解了"
)

CONTRAST_EMOTION_RE = re.compile(
    r"但|但是|可是|不过|然而|仍然|还是|依然"
)

CALM_EMOTION_RE = re.compile(
    r"不生气了|不担心了|已经消气|没事了|已经好了|问题解决了"
)

EMOTION_LEVEL_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_model(message: str) -> Optional[str]:
    match = MODEL_RE.search(message)
    return match.group(0).upper() if match else None


def detect_emotion(
    message: str,
) -> tuple[str, Literal["low", "medium", "high"]]:
    normalized = message.lower()

    if HIGH_EMOTION_RE.search(normalized):
        return "强烈不满", "high"

    if any(word in message for word in HIGH_EMOTION_WORDS):
        return "强烈不满", "high"

    if (
        POSITIVE_EMOTION_RE.search(normalized)
        and not CONTRAST_EMOTION_RE.search(normalized)
    ):
        return "满意或感谢", "low"

    if (
        CALM_EMOTION_RE.search(normalized)
        and not CONTRAST_EMOTION_RE.search(normalized)
    ):
        return "平稳", "low"

    for label, pattern in MEDIUM_EMOTION_CATEGORY_RULES:
        if pattern.search(normalized):
            return label, "medium"

    if any(word in message for word in MEDIUM_EMOTION_WORDS):
        return "不满或焦虑", "medium"

    if MEDIUM_EMOTION_RE.search(normalized):
        return "不满或焦虑", "medium"

    if re.search(
        r"才(?:买|用|使用)?了?[0-9一二三四五六七八九十]+"
        r"(?:天|小时|周).{0,8}(?:坏|裂|故障|不能|不行)",
        normalized,
    ):
        return "不满或焦虑", "medium"

    return "平稳", "low"


def update_case_understanding(
    case: CaseState,
    message: str,
) -> None:
    model = detect_model(message)

    if model:
        case.product_model = model

    current_emotion, current_level = detect_emotion(message)

    if (
        current_emotion in {"平稳", "满意或感谢"}
        and not CONTRAST_EMOTION_RE.search(message)
    ):
        case.emotion = current_emotion
        case.emotion_level = current_level
    else:
        recent_user_text = " ".join(
            turn.get("content", "")
            for turn in case.transcript[-8:]
            if turn.get("role") == "user"
        )
        recent_emotion, recent_level = detect_emotion(
            f"{recent_user_text} {message}"
        )

        if (
            EMOTION_LEVEL_RANK[recent_level]
            >= EMOTION_LEVEL_RANK[case.emotion_level]
        ):
            case.emotion = recent_emotion
            case.emotion_level = recent_level

    if OUTPUT_RE.search(message):
        case.intent = "无法给手机供电"

    elif CHARGE_RE.search(message):
        case.intent = "充电宝自身充电异常"

    if re.search(
        r"退换|退货|退款|退钱|退了|退掉|申请退|"
        r"换新|保修|质保|七天无理由",
        message,
    ):
        case.intent = "退换或质保咨询"

    if (
        case.intent
        and not case.symptoms
        and len(message.strip()) > 4
    ):
        case.symptoms.append(message[:300])


def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def init_db() -> None:
    connection = get_db()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            image_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_case(case: CaseState) -> None:
    data = model_to_dict(case)

    connection = get_db()

    connection.execute(
        """
        INSERT INTO cases (
            case_id,
            state_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(case_id)
        DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = excluded.updated_at
        """,
        (
            case.case_id,
            json.dumps(data, ensure_ascii=False),
            data["updated_at"],
            data["updated_at"],
        ),
    )

    connection.commit()
    connection.close()


def load_case(case_id: str) -> Optional[CaseState]:
    connection = get_db()

    row = connection.execute(
        """
        SELECT state_json
        FROM cases
        WHERE case_id = ?
        """,
        (case_id,),
    ).fetchone()

    connection.close()

    if not row:
        return None

    return CaseState(**json.loads(row["state_json"]))


def save_message(
    case_id: str,
    role: str,
    content: str,
) -> None:
    connection = get_db()

    connection.execute(
        """
        INSERT INTO messages (
            case_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            case_id,
            role,
            content,
            now(),
        ),
    )

    connection.commit()
    connection.close()


def save_image(
    case_id: str,
    image: dict,
) -> None:
    connection = get_db()

    connection.execute(
        """
        INSERT OR REPLACE INTO images (
            image_id,
            case_id,
            image_json,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            image["image_id"],
            case_id,
            json.dumps(image, ensure_ascii=False),
            now(),
        ),
    )

    connection.commit()
    connection.close()


init_db()


def new_case() -> CaseState:
    case = CaseState(
        case_id=f"CASE-{uuid.uuid4().hex[:10].upper()}",
        updated_at=now(),
    )

    CASES[case.case_id] = case
    save_case(case)

    return case


def get_case(case_id: Optional[str]) -> CaseState:
    if not case_id:
        return new_case()

    if case_id in CASES:
        return CASES[case_id]

    case = load_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="case_id 不存在",
        )

    CASES[case.case_id] = case
    return case


def add_turn(
    case: CaseState,
    role: str,
    content: str,
) -> None:
    message_time = now()

    case.transcript.append(
        {
            "role": role,
            "content": content,
            "at": message_time,
        }
    )

    case.updated_at = message_time

    save_message(
        case.case_id,
        role,
        content,
    )
    save_case(case)


OFFICIAL_SOURCES = [
    {
        "id": "support",
        "title": "Anker 官方支持中心",
        "url": "https://service.anker.com/?createTicket=true",
        "keywords": [
            "售后",
            "维修",
            "订单",
            "服务",
            "手册",
        ],
    },
    {
        "id": "warranty",
        "title": "Anker 官方保修与退款政策",
        "url": "https://www.anker.com/policies/refund-policy",
        "keywords": [
            "保修",
            "质保",
            "维修",
            "换新",
            "订单",
        ],
    },
    {
        "id": "safety",
        "title": "Anker 官方安全支持页面",
        "url": "https://service.anker.com/?createTicket=true",
        "keywords": [
            "安全",
            "鼓包",
            "发热",
            "过热",
            "冒烟",
            "异味",
            "漏液",
        ],
    },
    {
        "id": "a1647_recall",
        "title": "Anker A1647 官方召回公告",
        "url": "https://www.anker.com/eu-en/a1647-recall",
        "keywords": [
            "A1647",
            "召回",
            "序列号",
            "过热",
            "冒烟",
            "起火",
        ],
    },
]

ALLOWED_HOSTS = {
    "anker.com",
    "www.anker.com",
    "service.anker.com",
}

PAGE_CACHE: Dict[str, dict] = {}
EMBEDDING_CACHE: Dict[str, List[float]] = {}
CACHE_TTL_SECONDS = 1800

def load_knowledge_base() -> List[dict]:
    """加载 knowledge_base/powerbank_kb.json，文件缺失时返回空列表。"""
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"知识库文件不存在：{KNOWLEDGE_BASE_PATH}")
        return []
    try:
        data = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print("知识库格式错误：根节点必须是 JSON 数组")
            return []
        records = []
        for index, item in enumerate(data):
            if not isinstance(item, dict) or not item.get("content"):
                continue
            record = dict(item)
            record.setdefault("id", record.get("entry_id", f"kb-{index + 1}"))
            record.setdefault("title", "未命名知识片段")
            record.setdefault("keywords", [])
            records.append(record)
        print(f"已加载官方知识库 {len(records)} 条：{KNOWLEDGE_BASE_PATH}")
        return records
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取知识库失败：{exc}")
        return []


LOCAL_KB = load_knowledge_base()


def _vectorize(text: str) -> dict[str, float]:
    """纯 Python 的字符 n-gram 向量，避免额外依赖；适合中文短问题检索。"""
    text = re.sub(r"\s+", "", text.lower())
    grams = list(text)
    grams += [text[i:i + 2] for i in range(max(0, len(text) - 1))]
    vector = {}
    for gram in grams:
        vector[gram] = vector.get(gram, 0.0) + 1.0
    return vector


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)
    na = sum(value * value for value in a.values()) ** 0.5
    nb = sum(value * value for value in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embedding_client() -> Optional[OpenAI]:
    """可选的豆包 Embedding 客户端；未配置时返回 None。"""
    if os.getenv("DOUBAO_ENABLED", "false").lower() != "true":
        return None
    api_key = os.getenv("DOUBAO_API_KEY")
    base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    return OpenAI(api_key=api_key, base_url=base_url) if api_key else None


def _remote_embedding(text: str) -> Optional[List[float]]:
    """调用 Doubao-embedding-vision 多模态向量接口；失败时安全降级。"""
    model = os.getenv("DOUBAO_EMBEDDING_MODEL")
    client = _embedding_client()
    if not client or not model:
        return None
    cache_key = f"{model}:{text}"
    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    try:


        response = httpx.post(
            f"{os.getenv('DOUBAO_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3').rstrip('/')}/embeddings/multimodal",
            headers={
                "Authorization": f"Bearer {os.getenv('DOUBAO_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "instructions": (
                    "Target_modality: text. "
                    "Instruction: Represent the input for semantic retrieval. Query:"
                ),
                "input": [{"type": "text", "text": text[:6000]}],
                "encoding_format": "float",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if isinstance(data, list):
            vector = data[0].get("embedding")
        elif isinstance(data, dict):
            vector = data.get("embedding")
        else:
            vector = None
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedding Vision 未返回有效向量")
        vector = [float(value) for value in vector]
        EMBEDDING_CACHE[cache_key] = vector
        return vector
    except Exception as exc:
        print(f"Embedding 调用失败，改用本地向量：{exc}")
        return None


def _embedding_score(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def retrieve_local_kb(query: str, product_model: Optional[str] = None, top_k: int = 4) -> List[dict]:
    query_embedding = _remote_embedding(query)

    query_vector = _vectorize(query)
    results = []
    for item in LOCAL_KB:
        models = [
            str(model).strip()
            for model in item.get("product_models", [])
            if str(model).strip()
        ]
        exact_model_match = False
        generic_model_match = any(
            "通用" in model
            for model in models
        )

        if product_model:
            product_model_lower = product_model.lower()
            specific_models = [
                model
                for model in models
                if "通用" not in model
            ]
            exact_model_match = any(
                product_model_lower == model.lower()
                or product_model_lower in model.lower()
                or model.lower() in product_model_lower
                for model in specific_models
            )

            if (
                specific_models
                and not exact_model_match
            ):
                continue

        searchable = f"{item['title']} {item['content']} {' '.join(item['keywords'])}"
        item_embedding = _remote_embedding(searchable) if query_embedding is not None else None
        score = (
            _embedding_score(query_embedding, item_embedding)
            if query_embedding is not None and item_embedding is not None
            else _cosine(query_vector, _vectorize(searchable))
        )
        query_lower = query.lower()
        keyword_hits = sum(
            1
            for keyword in item["keywords"]
            if str(keyword).lower() in query_lower
        )
        score += keyword_hits * 0.08

        if exact_model_match:
            score += 0.45
        elif product_model and generic_model_match:
            score += 0.03

        if score < 0.18:
            continue

        if (
            product_model
            and not exact_model_match
            and keyword_hits == 0
        ):
            continue

        if score >= 0.18:
            results.append({
                **item,
                "url": item.get(
                    "source_url",
                    "local://anker-support-kb/" + item["id"],
                ),
                "source_type": "official_local_kb",
                "retrieval_method": "doubao_embedding" if query_embedding is not None else "local_ngram",
                "relevance_score": round(score, 4),
                "fetched_at": now(),
            })
    return sorted(results, key=lambda item: item["relevance_score"], reverse=True)[:top_k]


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    return (
        parsed.scheme == "https"
        and host in ALLOWED_HOSTS
    )


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "header",
            "footer",
            "nav",
        ]
    ):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)[:12000]


def fetch_official_page(
    source: dict,
) -> Optional[dict]:
    url = source["url"]

    if not allowed_url(url):
        return None

    cached = PAGE_CACHE.get(url)

    if cached:
        age = time.time() - cached["cached_at"]

        if age < CACHE_TTL_SECONDS:
            return cached["data"]

    try:
        with httpx.Client(
            timeout=10.0,
            follow_redirects=True,
            headers={
                "User-Agent": "AnkerAfterSalesAgent/0.5",
            },
        ) as client:
            response = client.get(url)
            response.raise_for_status()

        content = clean_html(response.text)

        if not content:
            return None

        data = {
            "id": source["id"],
            "title": source["title"],
            "source_type": "official_live",
            "content": content,
            "url": url,
            "fetched_at": now(),
        }

        PAGE_CACHE[url] = {
            "cached_at": time.time(),
            "data": data,
        }

        return data

    except Exception as exc:
        print(
            f"官方页面访问失败：{url}，原因：{exc}"
        )
        return None


def retrieve_official_sources(
    query: str,
    product_model: Optional[str] = None,
    top_k: int = 4,
) -> List[dict]:

    effective_query = strip_negated_risk_mentions(query)
    local_results = retrieve_local_kb(
        effective_query,
        product_model,
        top_k,
    )
    query_lower = effective_query.lower()
    candidates = []

    for source in OFFICIAL_SOURCES:
        score = 0
        lower_keywords = [
            keyword.lower()
            for keyword in source["keywords"]
        ]

        for keyword in lower_keywords:
            if keyword in query_lower:
                score += 1

        if (
            product_model
            and product_model.lower() in lower_keywords
        ):
            score += 3

        if RISK_RE.search(effective_query):
            if source["id"] in {
                "safety",
                "a1647_recall",
            }:
                score += 5

        if re.search(
            r"保修|质保|维修|订单|换新",
            effective_query,
        ):
            if source["id"] == "warranty":
                score += 4

        if score > 0:
            candidates.append(
                {
                    **source,
                    "score": score,
                }
            )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    results = []

    for source in candidates[:top_k]:
        page = fetch_official_page(source)

        if page:
            page["relevance_score"] = source["score"]
            results.append(page)


    merged = local_results + results
    unique = {}
    for item in merged:
        unique[item["id"]] = item
    return sorted(
        unique.values(),
        key=lambda item: item.get("relevance_score", 0),
        reverse=True,
    )[:top_k]


def load_work_orders() -> List[dict]:
    """读取脱敏/模拟历史工单；文件不存在或格式错误时安全降级。"""
    if not WORK_ORDER_PATH.exists():
        return []
    try:
        data = json.loads(WORK_ORDER_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取历史工单失败：{exc}")
        return []


WORK_ORDERS = load_work_orders()


def _case_text(item: dict) -> str:
    fields = (
        "产品类别", "产品型号", "故障类型", "用户描述",
        "处理结果", "处理步骤", "安全等级", "最终去向",
    )
    return " ".join(str(item.get(field, "")) for field in fields).lower()


def _fault_keywords(query: str) -> set[str]:
    words = re.findall(
        r"充不进电|无法充电|充电慢|充很久|指示灯不亮|不亮|发烫|发热|烫手|鼓包|异味|冒烟|电量不准|显示0%|放了很久|休眠|激活|输出异常|充不上",
        query.lower(),
    )
    return set(words)

def search_historical_cases(
    query: str,
    product_model: Optional[str] = None,
) -> dict:
    query_lower = query.lower()
    query_words = _fault_keywords(query)
    query_embedding = _remote_embedding(query) if WORK_ORDERS else None
    scored = []
    for item in WORK_ORDERS:

        category = str(item.get("产品类别", ""))
        if category and category != "充电宝":
            continue
        score = 0
        model = str(item.get("产品型号", "")).upper()
        if product_model and model == product_model.upper():
            score += 5
        text = _case_text(item)
        semantic_score = 0.0
        if query_embedding is not None:
            case_embedding = _remote_embedding(text)
            if case_embedding is not None:
                semantic_score = _embedding_score(query_embedding, case_embedding)
        if "充电宝" in text:
            score += 1
        score += sum(2 for word in query_words if word in text)
        query_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query_lower))
        case_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text))
        score += min(3, len(query_tokens & case_tokens))
        if str(item.get("是否解决", "")) == "是":
            score += 1

        if semantic_score >= 0.18 or score > 0:
            record = dict(item)
            record["匹配分"] = round(semantic_score, 4) if semantic_score else score
            record["向量相似度"] = round(semantic_score, 4)
            record["检索方式"] = "doubao_embedding" if semantic_score else "rule_fallback"
            record["数据来源"] = item.get("数据来源", "simulated")
            scored.append(record)
    scored.sort(key=lambda x: (x["向量相似度"], x["匹配分"]), reverse=True)
    return {
        "available": bool(WORK_ORDERS),
        "has_similar_cases": bool(scored),
        "cases": scored[:3],
        "message": "历史工单为脱敏模拟数据" if WORK_ORDERS else "未找到 work_orders.json",
    }


def doubao_client() -> Optional[OpenAI]:
    if (
        os.getenv("DOUBAO_ENABLED", "false").lower()
        != "true"
    ):
        return None

    api_key = os.getenv("DOUBAO_API_KEY")
    base_url = os.getenv(
        "DOUBAO_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/v3",
    )

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def doubao_reply(
    case: CaseState,
    message: str,
) -> Optional[str]:
    client = doubao_client()
    model = os.getenv("DOUBAO_MODEL")

    if not client or not model:
        return None

    official_context = "\n".join(
        [
            (
                f"{item['title']}\n"
                f"{item['content'][:3000]}\n"
                f"官方来源：{item['url']}"
            )
            for item in case.official_sources
        ]
    )

    history_context = "\n".join(
        [
            (
                f"工单号：{item.get('工单号', '未知')}；"
                f"型号：{item.get('产品型号', '未知')}；"
                f"描述：{item.get('用户描述', '')}；"
                f"处理结果：{item.get('处理结果', '')}；"
                f"处理步骤：{item.get('处理步骤', '')}；"
                f"数据来源：{item.get('数据来源', 'simulated')}"
            )
            for item in case.historical_cases
        ]
    )

    transcript_context = "\n".join(
        [
            (
                f"{'用户' if turn.get('role') == 'user' else '客服'}："
                f"{turn.get('content', '')}"
            )
            for turn in case.transcript[-10:]
        ]
    )

    prompt = f"""
你是 Anker 充电宝售后客服。

当前工单：
产品型号：{case.product_model or "未知"}
用户意图：{case.intent or "未知"}
用户情绪：{case.emotion or "平稳"}
用户症状：{case.symptoms}
证据：{case.evidence}
图片证据：{case.image_evidence}
风险：{case.risk_flags}
已完成步骤：{case.completed_steps}
下一步动作：{case.next_action}

官方资料：
{official_context or "暂无匹配的官方资料"}

历史案例（仅作相似处理路径参考，不得当作当前用户事实）：
{history_context or "暂无匹配历史案例"}

最近对话（必须结合上下文，已确认的信息不得重复询问）：
{transcript_context or "暂无历史对话"}

用户消息：
{message}

要求：
1. 只基于当前工单、官方资料和上面的历史案例回答；
2. 每次最多问两个问题；
3. 一次只建议一个可验证动作；
4. 不得编造订单、质保结果、召回结果或维修结论；
5. 如果官方资料不足，要明确说明无法确认；
6. 不得改变当前工单规定的下一步动作；
7. 不得把模拟历史工单描述成真实用户数据；
8. 图片证据为空时，不得声称看过图片或引用图片分析结果；
9. 风险列表为空时，不得声称检测到漏液、鼓包、发热等风险；
10. 用户没有真实订单和质保数据时，只能说明需要补充信息；
11. 必须利用最近对话，用户已经说明过的信息不得再次询问。
"""

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个安全、谨慎、专业的售后客服。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content
        return content.strip() if content else None

    except Exception as exc:
        print(f"豆包调用失败：{exc}")
        return None


def understanding_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])

    update_case_understanding(
        case,
        state["message"],
    )

    case.current_node = "understanding"
    case.updated_at = now()
    save_case(case)

    return {
        "current_node": "understanding",
        "route": "continue",
    }


def safety_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    message = state["message"]

    case.current_node = "safety_gate"

    risk_type = safety_risk_type(message)
    effective_message = strip_negated_risk_mentions(message)

    if (
        has_negated_risk_mention(message)
        and not risk_type
        and not RISK_RE.search(effective_message)
    ):
        remaining_risks = [
            flag
            for flag in case.risk_flags
            if not flag.endswith("：禁止继续DIY排障")
        ]

        if remaining_risks != case.risk_flags:
            case.risk_flags = remaining_risks

            if (
                not remaining_risks
                and case.status == "handoff"
                and case.handoff_reason
                and "触发安全转人工" in case.handoff_reason
            ):
                case.status = "diagnosing"
                case.resolution = None
                case.handoff_reason = None
                case.next_action = "继续确认故障现象"

    if risk_type:
        risk_flag = f"{risk_type}：禁止继续DIY排障"

        if risk_flag not in case.risk_flags:
            case.risk_flags.append(risk_flag)

        case.forbidden_actions = [
            "继续充电",
            "继续使用",
            "拆解设备",
            "挤压或刺穿设备",
        ]

        case.status = "handoff"
        case.resolution = "安全转人工"
        case.handoff_reason = f"检测到{risk_type}，触发安全转人工"
        case.next_action = "停止使用并联系人工售后"

        sources = retrieve_official_sources(
            message,
            case.product_model,
        )

        case.official_sources = sources
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                f"检测到{risk_type}，这属于安全风险。"
                "请立即停止使用和充电，不要拆解、挤压或刺穿设备。"
                "请将设备放在远离人员和可燃物的安全位置，"
                "并联系人工售后处理。"
            ),
            "action": "safety_handoff",
            "risk_level": "high",
            "citations": [
                item["title"]
                for item in sources
            ],
            "route": "finish",
            "current_node": "safety_gate",
        }

    if case.risk_flags:
        case.status = "handoff"
        case.next_action = "人工审核安全风险和退换资格"
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                "当前工单已经记录安全风险。"
                "出于安全原因，不能继续在线排障，"
                "也不能直接承诺退换。"
                "人工客服会结合图片证据、风险记录和购买信息处理。"
            ),
            "action": "safety_handoff",
            "risk_level": "high",
            "citations": [
                item["title"]
                for item in case.official_sources
            ],
            "route": "finish",
            "current_node": "safety_gate",
        }

    case.updated_at = now()
    save_case(case)

    return {
        "route": "continue",
        "current_node": "safety_gate",
    }


def emotion_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "emotion_check"

    if case.emotion_level == "high":
        case.status = "handoff"
        case.resolution = "优先人工接管"
        case.handoff_reason = "用户情绪等级较高"
        case.next_action = "人工客服优先接管"
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                "我理解这个问题确实让你很困扰。"
                "我已经记录当前工单，接下来优先安排人工处理，"
                "避免你重复描述。"
            ),
            "action": "priority_handoff",
            "risk_level": "none",
            "citations": [],
            "route": "finish",
            "current_node": "emotion_check",
        }

    case.updated_at = now()
    save_case(case)

    return {
        "route": "continue",
        "current_node": "emotion_check",
    }


def product_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "product_identification"

    if case.intent == "退换或质保咨询":
        case.updated_at = now()
        save_case(case)
        return {
            "route": "continue",
            "current_node": "product_identification",
        }

    if not case.product_model:
        case.next_action = "补充型号或上传型号标签照片"
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                "请查看充电宝底部或背面的型号标签，"
                "告诉我类似 A1647、A1289 或 A110B 的型号。"
                "如果不方便，也可以上传产品标签照片。"
            ),
            "action": "identify_product_model",
            "risk_level": "none",
            "citations": [],
            "route": "finish",
            "current_node": "product_identification",
        }

    case.updated_at = now()
    save_case(case)

    return {
        "route": "continue",
        "current_node": "product_identification",
    }


def image_evidence_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "image_evidence"

    needs_image = (
        case.intent
        in {
            "充电宝自身充电异常",
            "无法给手机供电",
        }
        and not case.photos_requested
        and not case.photos_received
    )

    if needs_image:
        case.photos_requested = True
        case.next_action = "上传产品和故障照片"
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                "为了确认型号和故障情况，请上传："
                "充电宝正面、底部型号标签、"
                "插入充电线时的指示灯和充电接口照片。"
                "如果有鼓包、破损、漏液或烧焦痕迹，"
                "也请拍照，但请先停止充电和使用。"
            ),
            "action": "request_photos",
            "risk_level": "none",
            "citations": [],
            "route": "finish",
            "current_node": "image_evidence",
        }

    case.updated_at = now()
    save_case(case)

    return {
        "route": "continue",
        "current_node": "image_evidence",
    }


def official_retrieval_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "official_retrieval"

    sources = retrieve_official_sources(
        state["message"],
        case.product_model,
    )

    case.official_sources = sources
    case.updated_at = now()
    save_case(case)

    return {
        "citations": [
            item["title"]
            for item in sources
        ],
        "current_node": "official_retrieval",
    }


def historical_case_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "historical_case_retrieval"

    result = search_historical_cases(
        state["message"],
        case.product_model,
    )

    case.historical_cases = result["cases"]
    case.has_similar_cases = result["has_similar_cases"]
    case.updated_at = now()
    save_case(case)

    return {
        "current_node": "historical_case_retrieval",
    }


def business_tool_node(state: SupportGraphState) -> SupportGraphState:
    """根据当前对话自动调用模拟订单、质保、序列号工具。"""
    case = get_case(state["case_id"])
    message = state["message"]
    case.current_node = "business_tools"
    results = []

    order_match = re.search(r"(?:SIM-)?ORDER[-_]?\d+|订单号[：:]?\s*([A-Za-z0-9_-]{5,})", message, re.IGNORECASE)
    serial_match = re.search(r"(?:SIM)?SN[A-Za-z0-9_-]{6,}|序列号[：:]?\s*([A-Za-z0-9_-]{6,})", message, re.IGNORECASE)

    if order_match:
        order_id = order_match.group(0).split("：")[-1].split(":")[-1].strip()
        results.append({"tool": "order_lookup", "result": lookup_order(order_id)})
    if serial_match:
        serial = serial_match.group(0).split("：")[-1].split(":")[-1].strip()
        results.append({"tool": "serial_lookup", "result": lookup_serial(serial)})

    if re.search(
        r"质保|保修|换新|退换|退货|退款|退钱|退了|"
        r"退掉|申请退|七天无理由|维修",
        message,
    ):
        order_id = order_match.group(0).split("：")[-1].split(":")[-1].strip() if order_match else None
        serial = serial_match.group(0).split("：")[-1].split(":")[-1].strip() if serial_match else None
        results.append({
            "tool": "warranty_lookup",
            "result": lookup_warranty(case.product_model or "未知", order_id=order_id, serial_number=serial),
        })
    if re.search(r"经销商|京东|天猫|亚马逊|线下购买", message):
        order_id = order_match.group(0).split("：")[-1].split(":")[-1].strip() if order_match else ""
        results.append({"tool": "dealer_order_lookup", "result": lookup_dealer_order(order_id)})

    if results:
        case.tool_results.extend(results)
        case.evidence.append("已调用模拟业务工具查询")
    case.updated_at = now()
    save_case(case)
    return {"current_node": "business_tools", "route": "continue"}


def diagnosis_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    message = state["message"]
    case.current_node = "diagnosis"

    citations = [
        item["title"]
        for item in case.official_sources
    ]

    warranty_request = re.search(
        r"退换|退货|退款|退钱|退了|退掉|申请退|"
        r"换新|保修|质保|七天无理由",
        message,
    )
    if warranty_request:
        case.status = "diagnosing"
        case.next_action = "核对订单号、购买日期和产品序列号"
        case.updated_at = now()
        save_case(case)
        return {
            "reply": (
                "我可以协助发起退换或质保申请，"
                "但不能直接承诺一定可以退换。"
                "请提供订单号、购买日期和产品序列号，"
                "系统核对后再决定是自助处理还是转人工审核。"
            ),
            "action": "warranty_guidance",
            "risk_level": "none",
            "citations": citations,
            "route": "generate",
            "current_node": "diagnosis",
        }

    resolved = re.search(
        r"已经好了|现在好了|可以了|恢复正常|已经恢复|能充电了|能正常充电|问题解决了|解决了",
        message,
    )
    if resolved and not re.search(r"没恢复|没有恢复|还是不行|仍然不行", message):
        case.status = "resolved"
        case.resolution = "用户反馈设备已恢复正常"
        case.next_action = "关闭工单并记录解决结果"
        case.updated_at = now()
        save_case(case)
        return {
            "reply": "太好了，已记录本次问题已经解决。工单状态已更新为已解决。",
            "action": "resolved",
            "risk_level": "none",
            "citations": citations,
            "route": "finish",
            "current_node": "diagnosis",
        }

    safety_confirmed = re.search(
        r"灯不亮|指示灯不亮|没有发热|不发热|"
        r"没有异味|没有鼓包|没有进水|无异常|正常",
        message,
    )

    failure_after_step = re.search(
        r"还是不行|仍然不行|没有恢复|没反应|依然不行",
        message,
    )

    if failure_after_step and case.completed_steps:
        case.status = "handoff"
        case.resolution = "转人工处理"
        case.handoff_reason = "基础排障后仍未恢复"
        case.next_action = "创建人工售后工单"
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                "设备经过基础排障后仍未恢复。"
                "我建议进入质保或维修人工流程。"
                "我已经整理好产品型号、故障现象和已尝试步骤，"
                "人工客服无需让你重复描述。"
            ),
            "action": "handoff",
            "risk_level": "none",
            "citations": citations,
            "route": "generate",
            "current_node": "diagnosis",
        }

    if safety_confirmed:
        evidence = "指示灯和基础安全状态已确认"

        if evidence not in case.evidence:
            case.evidence.append(evidence)

        step = "完成基础安全检查"

        if step not in case.completed_steps:
            case.completed_steps.append(step)

        case.next_action = (
            "更换正常线材和充电头，连接 10 秒"
        )
        case.updated_at = now()
        save_case(case)

        return {
            "reply": (
                "目前没有发现明显安全风险。"
                "请换一根确定正常的充电线和充电头，"
                "连接 10 秒后告诉我指示灯状态。"
            ),
            "action": "run_verified_step",
            "risk_level": "none",
            "citations": citations,
            "route": "generate",
            "current_node": "diagnosis",
        }

    case.status = "diagnosing"
    case.next_action = "确认指示灯和基础安全状态"
    case.updated_at = now()
    save_case(case)

    return {
        "reply": (
            "收到，我先确认一下："
            "插上电源后指示灯是否亮起？"
            "设备有没有异常发热、异味、"
            "鼓包、漏液或进水？"
        ),
        "action": "collect_evidence",
        "risk_level": "none",
        "citations": citations,
        "route": "generate",
        "current_node": "diagnosis",
    }


def response_generation_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "response_generation"

    fallback_reply = state.get(
        "reply",
        "抱歉，当前无法完成诊断。",
    )

    generated = None
    if state.get("action") != "warranty_guidance":
        generated = doubao_reply(
            case,
            state["message"],
        )

    case.updated_at = now()
    save_case(case)

    return {
        "reply": generated or fallback_reply,
        "current_node": "response_generation",
    }


def finish_node(
    state: SupportGraphState,
) -> SupportGraphState:
    case = get_case(state["case_id"])
    case.current_node = "finished"
    case.updated_at = now()
    save_case(case)

    return {
        "current_node": "finished",
    }


def route_after_gate(
    state: SupportGraphState,
) -> str:
    if state.get("route") == "finish":
        return "finish"

    return "continue"


def route_after_diagnosis(
    state: SupportGraphState,
) -> str:
    if state.get("risk_level") == "high":
        return "finish"

    return "generate"


def build_support_graph():
    builder = StateGraph(SupportGraphState)

    builder.add_node(
        "understanding",
        understanding_node,
    )
    builder.add_node(
        "safety_gate",
        safety_node,
    )
    builder.add_node(
        "emotion_check",
        emotion_node,
    )
    builder.add_node(
        "product_identification",
        product_node,
    )
    builder.add_node(
        "business_tools",
        business_tool_node,
    )
    builder.add_node(
        "image_evidence",
        image_evidence_node,
    )
    builder.add_node(
        "official_retrieval",
        official_retrieval_node,
    )
    builder.add_node(
        "historical_case_retrieval",
        historical_case_node,
    )
    builder.add_node(
        "diagnosis",
        diagnosis_node,
    )
    builder.add_node(
        "response_generation",
        response_generation_node,
    )
    builder.add_node(
        "finish",
        finish_node,
    )

    builder.add_edge(
        START,
        "understanding",
    )
    builder.add_edge(
        "understanding",
        "safety_gate",
    )

    builder.add_conditional_edges(
        "safety_gate",
        route_after_gate,
        {
            "finish": "finish",
            "continue": "emotion_check",
        },
    )

    builder.add_conditional_edges(
        "emotion_check",
        route_after_gate,
        {
            "finish": "finish",
            "continue": "product_identification",
        },
    )

    builder.add_conditional_edges(
        "product_identification",
        route_after_gate,
        {
            "finish": "finish",
            "continue": "business_tools",
        },
    )

    builder.add_edge(
        "business_tools",
        "image_evidence",
    )

    builder.add_conditional_edges(
        "image_evidence",
        route_after_gate,
        {
            "finish": "finish",
            "continue": "official_retrieval",
        },
    )

    builder.add_edge(
        "official_retrieval",
        "historical_case_retrieval",
    )
    builder.add_edge(
        "historical_case_retrieval",
        "diagnosis",
    )

    builder.add_conditional_edges(
        "diagnosis",
        route_after_diagnosis,
        {
            "finish": "finish",
            "generate": "response_generation",
        },
    )

    builder.add_edge(
        "response_generation",
        "finish",
    )
    builder.add_edge(
        "finish",
        END,
    )

    return builder.compile()


SUPPORT_GRAPH = build_support_graph()


def build_summary(case: CaseState) -> str:
    source_text = "\n".join(
        [
            f"- {item['title']}: {item['url']}"
            for item in case.official_sources
        ]
    )

    image_lines = []

    for image in case.image_evidence:
        analysis = image.get("analysis") or {}

        image_lines.append(
            "；".join(
                [
                    f"图片：{image.get('filename') or image['image_id']}",
                    f"型号：{analysis.get('model_name') or '未识别'}",
                    f"序列号：{analysis.get('serial_number') or '未识别'}",
                    f"指示灯：{analysis.get('indicator_status') or '未识别'}",
                    f"外观风险：{analysis.get('visible_risk') or '未识别'}",
                ]
            )
        )

    return "\n".join(
        [
            f"案件：{case.case_id}",
            f"状态：{case.status}",
            f"当前节点：{case.current_node}",
            f"产品类别：{case.product_category}",
            f"产品型号：{case.product_model or '待确认'}",
            f"意图：{case.intent or '待确认'}",
            f"情绪：{case.emotion or '未知'}",
            f"情绪等级：{case.emotion_level}",
            f"症状：{'；'.join(case.symptoms) or '待补充'}",
            f"风险：{'；'.join(case.risk_flags) or '未发现'}",
            f"证据：{'；'.join(case.evidence) or '无'}",
            f"图片证据：{' | '.join(image_lines) or '未上传'}",
            f"已完成：{'；'.join(case.completed_steps) or '无'}",
            f"接管原因：{case.handoff_reason or '客户请求人工'}",
            "历史案例：当前暂无可验证的真实历史工单数据",
            "官方资料：",
            source_text or "暂无匹配资料",
            "建议：人工客服继续确认订单、序列号、购买渠道和质保状态。",
        ]
    )


SIM_ORDERS = {
    "SIM-ORDER-1001": {
        "order_id": "SIM-ORDER-1001", "product_model": "A1687",
        "purchase_date": "2025-08-20", "channel": "官网",
        "status": "已完成"
    },
    "SIM-ORDER-1002": {
        "order_id": "SIM-ORDER-1002", "product_model": "A1647",
        "purchase_date": "2024-05-18", "channel": "京东",
        "status": "已完成"
    }
}

SIM_SERIALS = {
    "SIMSN16870001": {
        "serial_number": "SIMSN16870001", "product_model": "A1687",
        "recall_status": "未命中已知召回", "registered": True
    },
    "SIMSN16470001": {
        "serial_number": "SIMSN16470001", "product_model": "A1647",
        "recall_status": "需要根据官方页面进一步核验", "registered": True
    }
}


def lookup_order(order_id: str) -> dict:
    result = SIM_ORDERS.get(order_id.strip().upper())
    return {
        "found": bool(result),
        "data_source": "simulated",
        "message": "已找到模拟订单" if result else "未找到订单，请核对订单号或提供购买凭证",
        "order": result,
    }


def lookup_warranty(
    product_model: str,
    purchase_date: Optional[str] = None,
    order_id: Optional[str] = None,
    serial_number: Optional[str] = None,
) -> dict:
    order = SIM_ORDERS.get((order_id or "").strip().upper())
    model = product_model.strip().upper()
    if order and order.get("product_model") != model:
        return {"status": "待人工审核", "data_source": "simulated", "reason": "订单型号与用户提供型号不一致"}
    if not order and not purchase_date:
        return {"status": "待补充凭证", "data_source": "simulated", "reason": "需要订单号或购买日期"}
    return {
        "status": "待审核",
        "data_source": "simulated",
        "product_model": model,
        "order_id": order_id,
        "serial_number": serial_number,
        "reason": "模拟系统已接收信息，最终质保资格需要人工审核",
    }


def lookup_serial(serial_number: str) -> dict:
    result = SIM_SERIALS.get(serial_number.strip().upper())
    return {
        "found": bool(result),
        "data_source": "simulated",
        "message": "已找到模拟序列号" if result else "未找到序列号，请检查标签照片或转人工",
        "serial": result,
    }


def lookup_dealer_order(order_id: str, dealer: Optional[str] = None) -> dict:
    result = lookup_order(order_id)
    if result["found"] and dealer:
        result["order"]["dealer"] = dealer
    result["data_source"] = "simulated_dealer_system"
    result["message"] = "已在模拟经销商系统找到订单" if result["found"] else "模拟经销商系统未找到订单，请上传购买凭证"
    return result


@app.post("/v1/tools/order-lookup")
def order_lookup_tool(request: OrderLookupRequest) -> dict:
    return lookup_order(request.order_id)


@app.post("/v1/tools/warranty-lookup")
def warranty_lookup_tool(request: WarrantyLookupRequest) -> dict:
    return lookup_warranty(request.product_model, request.purchase_date, request.order_id, request.serial_number)


@app.post("/v1/tools/serial-lookup")
def serial_lookup_tool(request: SerialLookupRequest) -> dict:
    return lookup_serial(request.serial_number)


@app.post("/v1/tools/dealer-order-lookup")
def dealer_order_lookup_tool(request: DealerOrderLookupRequest) -> dict:
    return lookup_dealer_order(request.order_id, request.dealer)


def analyze_image_with_vision(content: bytes, content_type: str) -> Optional[dict]:
    """使用支持视觉输入的 DOUBAO_MODEL 分析故障图片。"""
    client = doubao_client()
    model = os.getenv("DOUBAO_MODEL")
    if not client or not model:
        return None
    image_data = base64.b64encode(content).decode("ascii")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "请分析这张充电宝售后故障照片，只输出JSON。字段必须包括："
                        "model_guess、model_confidence、visible_damage、bulging、"
                        "burn_mark、liquid_damage、indicator_status、risk_level、summary。"
                        "无法确认的字段填写unknown，不要臆测。"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{content_type};base64,{image_data}"
                    }},
                ],
            }],
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except Exception as exc:
        print(f"图片视觉分析失败：{exc}")
        return None


def vision_flag_is_true(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {
        "true",
        "yes",
        "1",
        "是",
        "有",
    }


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "anker-power-bank-support",
        "version": app.version,
    }


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/")


@app.post("/v1/cases", response_model=CaseState)
def create_case() -> CaseState:
    return new_case()


@app.get("/v1/cases/{case_id}", response_model=CaseState)
def read_case(case_id: str) -> CaseState:
    return get_case(case_id)


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    case = get_case(request.case_id)

    add_turn(
        case,
        "user",
        request.message,
    )

    initial_state: SupportGraphState = {
        "case_id": case.case_id,
        "message": request.message,
        "reply": "",
        "action": "unknown",
        "risk_level": "none",
        "citations": [],
        "route": "continue",
        "current_node": "start",
    }

    result = SUPPORT_GRAPH.invoke(
        initial_state
    )

    reply = result.get(
        "reply",
        "抱歉，当前无法完成诊断，请联系人工客服。",
    )

    action = result.get(
        "action",
        "handoff",
    )

    risk_level = result.get(
        "risk_level",
        "none",
    )

    citations = result.get(
        "citations",
        [],
    )

    case = get_case(case.case_id)

    add_turn(
        case,
        "assistant",
        reply,
    )

    save_case(case)

    return ChatResponse(
        case=case,
        reply=reply,
        action=action,
        risk_level=risk_level,
        citations=citations,
    )


@app.post("/v1/cases/{case_id}/images")
async def upload_image(
    case_id: str,
    file: UploadFile = File(...),
) -> dict:
    case = get_case(case_id)

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="只支持 JPG、PNG 或 WEBP 图片",
        )

    content = await file.read()

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="图片不能超过 8MB",
        )

    extension = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[file.content_type]

    image_id = uuid.uuid4().hex

    image_path = UPLOAD_DIR / (
        f"{case.case_id}-{image_id}{extension}"
    )

    image_path.write_bytes(content)

    record = {
        "image_id": image_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "path": str(image_path),
        "analysis_status": "uploaded_not_analyzed",
        "analysis": None,
        "created_at": now(),
    }

    vision_result = analyze_image_with_vision(content, file.content_type)
    if vision_result:
        record["analysis_status"] = "analyzed"
        record["analysis"] = vision_result
        case.image_evidence.append(record)
        case.evidence.append("视觉模型已完成图片分析")
        guessed_model = str(vision_result.get("model_guess", ""))
        if guessed_model and guessed_model.lower() != "unknown":
            detected = detect_model(guessed_model) or guessed_model.upper()
            case.product_model = detected
        risk_level = str(vision_result.get("risk_level", "")).lower()
        if risk_level == "high" or any(
            vision_flag_is_true(vision_result.get(key))
            for key in ("bulging", "burn_mark", "liquid_damage")
        ):
            risk_flag = "视觉模型识别到潜在设备安全风险"
            if risk_flag not in case.risk_flags:
                case.risk_flags.append(risk_flag)
            case.forbidden_actions = ["继续充电", "继续使用", "拆解设备", "挤压或刺穿设备"]
            case.status = "handoff"
            case.resolution = "图片安全风险转人工"
            case.handoff_reason = "视觉模型识别到潜在设备安全风险"
            case.next_action = "停止使用并联系人工售后"

    if not vision_result:
        case.image_evidence.append(record)

    case.photos_received = True
    if case.status != "handoff":
        case.next_action = "分析图片中的型号、指示灯和外观风险"
    case.evidence.append(
        f"已上传图片：{file.filename or image_id}"
    )
    case.updated_at = now()

    save_image(
        case.case_id,
        record,
    )
    save_case(case)

    return {
        "ok": True,
        "case_id": case.case_id,
        "image": record,
        "next_action": case.next_action,
    }


@app.post("/v1/cases/{case_id}/images/{image_id}/analysis")
def save_image_analysis(
    case_id: str,
    image_id: str,
    request: ImageAnalysisRequest,
) -> dict:
    case = get_case(case_id)

    target = None

    for image in case.image_evidence:
        if image["image_id"] == image_id:
            target = image
            break

    if target is None:
        raise HTTPException(
            status_code=404,
            detail="图片不存在",
        )

    analysis = {
        "model_name": request.model_name,
        "serial_number": request.serial_number,
        "indicator_status": request.indicator_status,
        "visible_risk": request.visible_risk,
        "description": request.description,
        "analyzed_at": now(),
    }

    target["analysis_status"] = "analyzed"
    target["analysis"] = analysis

    if request.model_name:
        case.product_model = request.model_name.upper()
        case.evidence.append(
            f"图片识别到型号：{request.model_name}"
        )

    if request.serial_number:
        case.evidence.append(
            f"图片识别到序列号：{request.serial_number}"
        )

    if request.indicator_status:
        case.evidence.append(
            f"图片识别到指示灯状态：{request.indicator_status}"
        )

    if request.visible_risk:
        case.evidence.append(
            f"图片识别到外观风险：{request.visible_risk}"
        )

        if RISK_RE.search(request.visible_risk):
            risk_flag = "图片识别到潜在安全风险"

            if risk_flag not in case.risk_flags:
                case.risk_flags.append(risk_flag)

            case.status = "handoff"
            case.resolution = "图片风险转人工"
            case.handoff_reason = "图片中存在潜在设备安全风险"
            case.next_action = "停止使用并联系人工售后"

    case.updated_at = now()

    save_image(
        case.case_id,
        target,
    )
    save_case(case)

    return {
        "ok": True,
        "case_id": case.case_id,
        "image_id": image_id,
        "case": case,
    }


@app.post(
    "/v1/cases/{case_id}/handoff",
    response_model=HandoffResponse,
)
def handoff(
    case_id: str,
) -> HandoffResponse:
    case = get_case(case_id)

    case.status = "handoff"
    case.resolution = case.resolution or "人工接管"
    case.handoff_reason = (
        case.handoff_reason
        or "客户或系统请求人工处理"
    )
    case.next_action = "人工客服查看接管摘要"
    case.updated_at = now()

    save_case(case)

    return HandoffResponse(
        case_id=case.case_id,
        summary=build_summary(case),
        status="handoff",
    )
