import datetime
import operator
from typing import Annotated, Sequence, TypedDict
import json

from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env файла
load_dotenv()

# Проверка наличия GOOGLE_API_KEY при загрузке
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("Переменная окружения GOOGLE_API_KEY не установлена. Пожалуйста, установите ее.")

# 1. Определение инструмента
@tool
def get_current_time() -> dict:
    """Возвращает текущее время UTC в формате ISO‑8601.
    Пример ответа → {"utc": "2025‑05‑21T06:42:00Z"}"""
    # Используем utcnow() и isoformat() для получения времени в формате ISO-8601
    # с "Z" для обозначения UTC (Zulu time)
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}


# 2. Определение состояния графа
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 3. Определение функции агента
class Agent:
    """
    Агент, использующий модель Gemini и набор инструментов для обработки сообщений.
    """
    def __init__(self, llm_model_name: str = "gemini-2.0-flash-001"): # Использование модели Gemini
        # Инициализируем LLM с доступом к инструменту
        self.tools = [get_current_time]
        # Используем ChatGoogleGenerativeAI для Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model_name, temperature=0
        ).bind_tools(self.tools)

    def run(self, state: AgentState):
        current_messages = state["messages"]
        response = self.llm.invoke(current_messages)
        # Если LLM решил вызвать инструмент, добавляем вызов инструмента в сообщения
        if response.tool_calls:
            tool_messages = []
            for tool_call_data in response.tool_calls: 
                
                tool_name = tool_call_data.get('name')
                tool_args = tool_call_data.get('args', {}) # По умолчанию пустой словарь, если 'args' отсутствует
                tool_id = tool_call_data.get('id') # Это будет tool_call_id для FunctionMessage

                # Проверка наличия основных данных
                if not tool_name or not tool_id:
                    error_msg = f"Некорректные данные вызова инструмента (отсутствует имя или id): {tool_call_data}"
                    print(error_msg)
                    # Используем заглушки, если критически важная информация отсутствует для FunctionMessage
                    fn_name = tool_name if tool_name else "unknown_tool_name_in_error"
                    fn_tool_call_id = tool_id if tool_id else "unknown_tool_call_id_in_error"
                    tool_messages.append(FunctionMessage(
                        content=error_msg,
                        name=fn_name,
                        tool_call_id=fn_tool_call_id
                    ))
                    continue

                try:
                    tool_function = globals().get(tool_name)
                    if callable(tool_function):
                         # Вызываем функцию, декорированную @tool
                         tool_output = tool_function.invoke(tool_args)
                         tool_messages.append(FunctionMessage(
                            content=json.dumps(tool_output),
                            name=tool_name, # Имя вызванной функции
                            tool_call_id=tool_id # ID исходного вызова инструмента из AIMessage
                        ))
                    else:
                         # Обработка случая, когда инструмент не найден
                         error_message = f"Ошибка: Инструмент с именем '{tool_name}' не найден или не является функцией."
                         tool_messages.append(FunctionMessage(content=error_message, name=tool_name, tool_call_id=tool_id))
                         print(error_message) # Логирование ошибки

                except Exception as e:
                    # Обработка любых других ошибок при выполнении инструмента
                    error_message = f"Ошибка при выполнении инструмента {tool_name}: {e}"
                    tool_messages.append(FunctionMessage(content=error_message, name=tool_name, tool_call_id=tool_id))
                    print(error_message) # Логирование ошибки

            return {"messages": [response] + tool_messages}
        else:
            # Если LLM ответил напрямую, просто возвращаем его ответ
            return {"messages": [response]}

# 4. Построение графа
def create_graph():
    workflow = StateGraph(AgentState)

    agent_node = Agent()

    workflow.add_node("agent", agent_node.run)

    workflow.set_entry_point("agent")

    # Определяем функцию условной логики
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        # Если последнее сообщение - FunctionMessage, значит, инструмент только что был запущен.
        # Агент должен обработать вывод инструмента.
        if isinstance(last_message, FunctionMessage):
            return "agent"
        # Если последнее сообщение - AIMessage:
        elif isinstance(last_message, AIMessage):
            # Если у него есть tool_calls, агент должен их выполнить.
            # (Agent.run уже делает это, так что этот путь означает цикл для обработки результатов
            # или если инструменты почему-то еще не были запущены)
            if last_message.tool_calls:
                return "agent"
            # В противном случае, если нет tool_calls, это окончательный ответ.
            else:
                return END
        # По умолчанию END для любого другого неожиданного типа сообщения в качестве последнего сообщения.
        return END

    # Добавляем условный край: если есть вызовы инструментов, запускаем агент снова
    # (для обработки вывода инструмента), иначе завершаем
    workflow.add_conditional_edges("agent", should_continue)
    
    app = workflow.compile()
    return app

# Запуск приложения (для `langgraph dev`)
app = create_graph()

if __name__ == "__main__":
    # Пример использования (для тестирования напрямую, без `langgraph dev`)
    # Проверка переменной окружения уже выполняется при загрузке .env
    
    print("Убедитесь, что у вас установлена переменная окружения GOOGLE_API_KEY")
    print("Приложение готово к запуску с помощью `langgraph dev`")