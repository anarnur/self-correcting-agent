# Вместо старой строки: model = ChatOpenAI(...)
# Пишем вот так:

model = ChatOpenAI(
    model="grok-2-1212",  # Или "grok-2", если есть доступ
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.x.ai/v1",
    temperature=0.7
)