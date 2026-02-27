import os
import datetime
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# 1. Описание памяти (состояния) нашего агента
class AgentState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int

# 2. Подключаем Ollama (убедись, что программа Ollama запущена)
model = ChatOllama(model="llama3", temperature=0.7)

# --- УЗЛЫ ГРАФА (ЛОГИКА) ---

def writer_node(state: AgentState):
    print("--- ПИСАТЕЛЬ: Создаю текст на русском ---")
    # Добавляем жесткое условие про язык в промпт
    prompt = f"Напиши пост для блога НА РУССКОМ ЯЗЫКЕ на тему: {state['task']}. Текст должен быть полезным и интересным."
    res = model.invoke(prompt)
    return {"draft": res.content, "revision_count": state.get("revision_count", 0) + 1}

def critic_node(state: AgentState):
    print("--- КРИТИК: Анализирую и ищу ошибки ---")
    prompt = (
        f"Проверь этот текст: {state['draft']}\n\n"
        "Твоя задача: если текст качественный и написан на русском, ответь только одним словом 'ГОТОВО'. "
        "Если в тексте есть ошибки, английские слова или он скучный — напиши список правок НА РУССКОМ."
    )
    res = model.invoke(prompt)
    return {"critique": res.content}

def editor_node(state: AgentState):
    print("--- РЕДАКТОР: Вношу правки ---")
    prompt = (
        f"Оригинал текста: {state['draft']}\n"
        f"Правки критика: {state['critique']}\n"
        "Перепиши текст полностью на русском языке, исправив все замечания."
    )
    res = model.invoke(prompt)
    return {"draft": res.content, "revision_count": state["revision_count"] + 1}

# --- ЛОГИКА ПЕРЕХОДОВ (МАРШРУТИЗАЦИЯ) ---

def decide_to_finish(state: AgentState):
    # Если критик доволен или мы превысили лимит правок (3 раза)
    if "ГОТОВО" in state["critique"].upper() or state["revision_count"] >= 3:
        return "end"
    return "edit"

# --- СБОРКА АГЕНТА (ГРАФА) ---

workflow = StateGraph(AgentState)

workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("editor", editor_node)

workflow.set_entry_point("writer")
workflow.add_edge("writer", "critic")

workflow.add_conditional_edges(
    "critic",
    decide_to_finish,
    {
        "edit": "editor",
        "end": END
    }
)

workflow.add_edge("editor", "critic")

app = workflow.compile()

# --- ЗАПУСК ---

if __name__ == "__main__":
    # Задаем тему
    task_input = {
        "task": "Почему нейросети не заменят людей, а станут их помощниками?", 
        "revision_count": 0
    }
    
    print("🚀 Запуск агента...")
    final_result = app.invoke(task_input)
    
    # Сохраняем финальный результат
    with open("report.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- РУССКИЙ ОТЧЕТ {datetime.datetime.now()} ---\n")
        f.write(final_result["draft"])
        f.write("\n" + "="*50 + "\n")
    
    print("\n✅ Готово! Результат на русском языке добавлен в report.txt")

