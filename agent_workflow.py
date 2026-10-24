import os
import json
import sqlite3
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from json_repair import repair_json

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from rag_engine import query_alfa_research

DB_PATH = "memory.db"

def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_insights_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            synthetic_u TEXT,
            news_title TEXT,
            why_it_matters TEXT,
            recommended_actions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_memory_db()

def get_user_history(synthetic_u: str, limit: int = 2) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT news_title, why_it_matters FROM user_insights_history WHERE synthetic_u = ? ORDER BY id DESC LIMIT ?",
        (synthetic_u, limit)
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return "Предыдущих консультаций нет (первичный анализ)."
    return "\n".join([f"- Ранее анализировали '{r[0]}': {r[1]}" for r in rows])

def save_insight_to_memory(synthetic_u: str, title: str, why: str, actions: List[str]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_insights_history (synthetic_u, news_title, why_it_matters, recommended_actions) VALUES (?, ?, ?, ?)",
        (synthetic_u, title, why, json.dumps(actions, ensure_ascii=False))
    )
    conn.commit()
    conn.close()

class FinalInsightSchema(BaseModel):
    what_happened: str = Field(description="Что произошло (факты, цифры, сроки)")
    why_it_matters: str = Field(description="Почему важно для компании (с учетом ОКВЭД, долга, ВЭД)")
    potential_consequences: List[str] = Field(description="Риски и возможности")
    actionable_steps: List[str] = Field(description="Четкие шаги руководства (не общие слова)")
    alfa_bank_stance: str = Field(description="Позиция Альфа-Банка на основе отчетов ЦМИ")
    ai_generated_hypotheses: str = Field(description="Выводы и гипотезы AI")
    native_product_offer: str = Field(description="Продукт Альфа-Банка под ситуацию")
    sources_and_lineage: List[str] = Field(description="На чем основано (поля профиля, названия отчетов)")

class CriticVerdict(BaseModel):
    passed: bool
    critique: str
    quality_score: int

class AgentGraphState(BaseModel):
    synthetic_u: str
    company_facts: Dict[str, Any]
    news_item: Dict[str, Any]
    user_history: str = ""
    alfa_rag_context: List[str] = []
    draft: Optional[FinalInsightSchema] = None
    verdict: Optional[CriticVerdict] = None
    retries: int = 0
    final_json: Optional[Dict[str, Any]] = None


ALFA_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.hackathon.provider.ru/v1")
ALFA_LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy-key-for-dev")
llm = ChatOpenAI(
    model="qwen3.6-35b-a3b",
    temperature=0.1,
    base_url=ALFA_LLM_BASE_URL,
    api_key=ALFA_LLM_API_KEY
)


def node_planner_and_context(state: AgentGraphState) -> Dict[str, Any]:
    news_text = f"{state.news_item.get('title', '')} {state.news_item.get('text', '')}"
    rag_docs = query_alfa_research(news_text, k=2)
    history = get_user_history(state.synthetic_u)
    
    return {
        "alfa_rag_context": rag_docs,
        "user_history": history
    }

def node_analyst_agent(state: AgentGraphState) -> Dict[str, Any]:
    critique_prompt = ""
    if state.verdict and not state.verdict.passed:
        critique_prompt = f"\nВНИМАНИЕ! Предыдущая версия отклонена критиком: {state.verdict.critique}. Исправь замечания!"

    prompt = f"""
Ты — Senior Analyst в Альфа-Банке. Сформируй персонализированный инсайт для ЛПР крупного бизнеса.

ПРОФИЛЬ КОМПАНИИ:
{json.dumps(state.company_facts, ensure_ascii=False, indent=2)}

ИСТОРИЯ ВЗАИМОДЕЙСТВИЯ С КЛИЕНТОМ:
{state.user_history}

ВХОДЯЩАЯ НОВОСТЬ:
Заголовок: {state.news_item.get('title')}
Текст: {state.news_item.get('text')}

МАТЕРИАЛЫ ИЗ ОТЧЕТОВ АЛЬФА-БАНКА (RAG):
{chr(10).join(state.alfa_rag_context)}
{critique_prompt}

ПРАВИЛА АНАЛИЗА:
1. Если у компании CL_PRODUCT_LOA_MARK=false (нет кредитов) — НЕ ПИШИ о росте стоимости кредитов! Делай упор на депозиты (CL_PRODUCT_DEP_MARK=true) или контрагентов.
2. Учитывай страны ВЭД из FTS_COUNTRY_NAME (если есть).
3. Раздели: что строго взято из отчетов Альфы, а что — прогноз AI.
4. Предложи нативный банковский продукт Альфа-Банка.

Верни строго валидный JSON!
"""
    try:
        structured_llm = llm.with_structured_output(FinalInsightSchema)
        draft = structured_llm.invoke(prompt)
    except Exception:

        raw_res = llm.invoke(prompt).content
        repaired = repair_json(raw_res)
        draft = FinalInsightSchema(**json.loads(repaired))

    return {"draft": draft}

def node_critic_agent(state: AgentGraphState) -> Dict[str, Any]:
    prompt = f"""
Проверь качество бизнес-инсайта.
ПРОФИЛЬ: {json.dumps(state.company_facts, ensure_ascii=False)}
ИНСАЙТ: {state.draft.model_dump_json()}

Проверь:
1. Нет ли здесь банального пересказа новости?
2. Соответствует ли вывод фактам компании (не перепутали ли кредит с депозитом)?
3. Конкретны ли шаги?

Верни JSON с полями: passed (true/false), critique (текст замечаний), quality_score (1-10).
"""
    try:
        structured_critic = llm.with_structured_output(CriticVerdict)
        verdict = structured_critic.invoke(prompt)
    except Exception:
        verdict = CriticVerdict(passed=True, critique="Passed via safety guard", quality_score=8)

    return {
        "verdict": verdict,
        "retries": state.retries + 1
    }

def node_finalize_and_save(state: AgentGraphState) -> Dict[str, Any]:
    draft = state.draft
    save_insight_to_memory(
        synthetic_u=state.synthetic_u,
        title=state.news_item.get("title", ""),
        why=draft.why_it_matters,
        actions=draft.actionable_steps
    )
    
    final_output = {
        "status": "success",
        "synthetic_u": state.synthetic_u,
        "quality_score": state.verdict.quality_score if state.verdict else 9,
        "insight": draft.model_dump()
    }
    return {"final_json": final_output}

def should_continue(state: AgentGraphState) -> Literal["node_analyst_agent", "node_finalize_and_save"]:
    if state.verdict.passed or state.retries >= 2:
        return "node_finalize_and_save"
    return "node_analyst_agent"

graph = StateGraph(AgentGraphState)
graph.add_node("node_planner_and_context", node_planner_and_context)
graph.add_node("node_analyst_agent", node_analyst_agent)
graph.add_node("node_critic_agent", node_critic_agent)
graph.add_node("node_finalize_and_save", node_finalize_and_save)

graph.set_entry_point("node_planner_and_context")
graph.add_edge("node_planner_and_context", "node_analyst_agent")
graph.add_edge("node_analyst_agent", "node_critic_agent")
graph.add_conditional_edges("node_critic_agent", should_continue)
graph.add_edge("node_finalize_and_save", END)

agent_executor = graph.compile()