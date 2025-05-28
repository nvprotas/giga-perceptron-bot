import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import os
from dataclasses import dataclass

import telebot
from telebot import types
from telebot.async_telebot import AsyncTeleBot
import threading
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode


import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('health_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Создаем бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Словарь для хранения состояний пользователей
user_states = {}


@dataclass
class UserHealthState:
    """Класс для хранения состояния здоровья пользователя"""
    user_id: int
    
    # Физические показатели
    water_intake: float = 0.0
    steps_count: int = 0
    exercise_minutes: int = 0
    sleep_hours: float = 0.0
    
    # Ментальные показатели
    stress_level: int = 5
    mood_score: int = 5
    meditation_minutes: int = 0
    social_interactions: int = 0
    
    # Рабочие показатели
    work_hours: float = 0.0
    breaks_taken: int = 0
    tasks_completed: List[str] = None
    meetings_attended: int = 0
    
    # История и активности
    daily_activities: List[Dict] = None
    achievements: List[str] = None
    challenges: List[str] = None
    
    def __post_init__(self):
        if self.tasks_completed is None:
            self.tasks_completed = []
        if self.daily_activities is None:
            self.daily_activities = []
        if self.achievements is None:
            self.achievements = []
        if self.challenges is None:
            self.challenges = []
    
    def add_activity(self, activity: str, category: str):
        """Добавляет активность в дневной журнал"""
        self.daily_activities.append({
            "activity": activity,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"User {self.user_id}: Added activity - {activity}")
    
    def get_health_summary(self) -> Dict:
        """Возвращает сводку по здоровью"""
        return {
            "physical": {
                "water_intake": self.water_intake,
                "steps_count": self.steps_count,
                "exercise_minutes": self.exercise_minutes,
                "sleep_hours": self.sleep_hours
            },
            "mental": {
                "stress_level": self.stress_level,
                "mood_score": self.mood_score,
                "meditation_minutes": self.meditation_minutes,
                "social_interactions": self.social_interactions
            },
            "work": {
                "work_hours": self.work_hours,
                "breaks_taken": self.breaks_taken,
                "tasks_completed": len(self.tasks_completed),
                "meetings_attended": self.meetings_attended
            }
        }
    
    def get_daily_summary_text(self) -> str:
        """Возвращает текстовую сводку дня для пользователя"""
        summary = self.get_health_summary()
        
        text = f"📊 *Ваша сводка за сегодня:*\n\n"
        
        # Физические показатели
        text += f"💪 *Физическое здоровье:*\n"
        text += f"💧 Вода: {summary['physical']['water_intake']}л\n"
        text += f"👣 Шаги: {summary['physical']['steps_count']}\n"
        text += f"🏃 Упражнения: {summary['physical']['exercise_minutes']} мин\n"
        text += f"😴 Сон: {summary['physical']['sleep_hours']} ч\n\n"
        
        # Ментальные показатели
        text += f"🧠 *Ментальное состояние:*\n"
        text += f"😊 Настроение: {summary['mental']['mood_score']}/10\n"
        text += f"😰 Стресс: {summary['mental']['stress_level']}/10\n"
        text += f"🧘 Медитация: {summary['mental']['meditation_minutes']} мин\n"
        text += f"👥 Соц. контакты: {summary['mental']['social_interactions']}\n\n"
        
        # Рабочие показатели
        text += f"💼 *Работа:*\n"
        text += f"⏰ Рабочих часов: {summary['work']['work_hours']}\n"
        text += f"☕ Перерывов: {summary['work']['breaks_taken']}\n"
        text += f"✅ Задач выполнено: {summary['work']['tasks_completed']}\n"
        text += f"🤝 Встреч: {summary['work']['meetings_attended']}\n"
        
        return text


def get_user_state(user_id: int) -> UserHealthState:
    """Получает или создает состояние пользователя"""
    if user_id not in user_states:
        user_states[user_id] = UserHealthState(user_id=user_id)
        logger.info(f"Created new user state for user {user_id}")
    return user_states[user_id]


# Инструменты для работы с конкретным пользователем
def create_user_tools(user_id: int):
    """Создает инструменты для конкретного пользователя"""
    user_state = get_user_state(user_id)
    
    @tool
    def log_water_intake(amount: float) -> str:
        """Записывает потребление воды в литрах"""
        user_state.water_intake += amount
        user_state.add_activity(f"Выпил {amount}л воды", "physical")
        return f"Записано потребление воды: {amount}л. Всего за день: {user_state.water_intake}л"

    @tool
    def log_exercise(minutes: int, exercise_type: str) -> str:
        """Записывает физическую активность"""
        user_state.exercise_minutes += minutes
        user_state.add_activity(f"{exercise_type} - {minutes} минут", "physical")
        return f"Записана активность: {exercise_type} ({minutes} мин). Всего упражнений: {user_state.exercise_minutes} мин"

    @tool
    def log_steps(steps: int) -> str:
        """Записывает количество шагов"""
        user_state.steps_count += steps
        user_state.add_activity(f"Прошел {steps} шагов", "physical")
        return f"Записано шагов: {steps}. Всего за день: {user_state.steps_count}"

    @tool
    def log_sleep(hours: float) -> str:
        """Записывает продолжительность сна"""
        user_state.sleep_hours = hours
        user_state.add_activity(f"Спал {hours} часов", "physical")
        return f"Записан сон: {hours} часов"

    @tool
    def update_mood(score: int) -> str:
        """Обновляет оценку настроения (1-10)"""
        user_state.mood_score = max(1, min(10, score))
        user_state.add_activity(f"Настроение: {score}/10", "mental")
        return f"Настроение обновлено: {score}/10"

    @tool
    def update_stress(level: int) -> str:
        """Обновляет уровень стресса (1-10)"""
        user_state.stress_level = max(1, min(10, level))
        user_state.add_activity(f"Уровень стресса: {level}/10", "mental")
        return f"Уровень стресса обновлен: {level}/10"

    @tool
    def log_meditation(minutes: int) -> str:
        """Записывает время медитации"""
        user_state.meditation_minutes += minutes
        user_state.add_activity(f"Медитация {minutes} минут", "mental")
        return f"Записана медитация: {minutes} мин. Всего: {user_state.meditation_minutes} мин"

    @tool
    def log_social_interaction(description: str) -> str:
        """Записывает социальное взаимодействие"""
        user_state.social_interactions += 1
        user_state.add_activity(f"Социальное взаимодействие: {description}", "mental")
        return f"Записано социальное взаимодействие: {description}"

    @tool
    def log_work_hours(hours: float) -> str:
        """Записывает рабочие часы"""
        user_state.work_hours += hours
        user_state.add_activity(f"Работал {hours} часов", "work")
        return f"Записано рабочих часов: {hours}. Всего: {user_state.work_hours}"

    @tool
    def log_break(duration: int) -> str:
        """Записывает перерыв"""
        user_state.breaks_taken += 1
        user_state.add_activity(f"Перерыв {duration} минут", "work")
        return f"Записан перерыв: {duration} мин. Всего перерывов: {user_state.breaks_taken}"

    @tool
    def complete_task(task_name: str) -> str:
        """Отмечает задачу как выполненную"""
        user_state.tasks_completed.append(task_name)
        user_state.add_activity(f"Выполнена задача: {task_name}", "work")
        return f"Задача выполнена: {task_name}. Всего задач: {len(user_state.tasks_completed)}"

    @tool
    def log_meeting(meeting_name: str, duration: int) -> str:
        """Записывает участие во встрече"""
        user_state.meetings_attended += 1
        user_state.add_activity(f"Встреча: {meeting_name} ({duration} мин)", "work")
        return f"Записана встреча: {meeting_name} ({duration} мин)"

    @tool
    def add_achievement(achievement: str) -> str:
        """Добавляет достижение дня"""
        user_state.achievements.append(achievement)
        user_state.add_activity(f"Достижение: {achievement}", "summary")
        return f"Добавлено достижение: {achievement}"

    @tool
    def add_challenge(challenge: str) -> str:
        """Добавляет вызов/сложность дня"""
        user_state.challenges.append(challenge)
        user_state.add_activity(f"Вызов: {challenge}", "summary")
        return f"Добавлен вызов: {challenge}"

    return {
        'health_tools': [log_water_intake, log_exercise, log_steps, log_sleep],
        'mental_health_tools': [update_mood, update_stress, log_meditation, log_social_interaction],
        'schedule_tools': [log_work_hours, log_break, complete_task, log_meeting],
        'summary_tools': [add_achievement, add_challenge]
    }


# LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini", 
    temperature=0.7, 
    streaming=False,
    api_key=OPENAI_API_KEY, 
    max_completion_tokens=1024
)


# Системные шаблоны (адаптированные для Telegram)
ROUTER_SYSTEM_TEMPLATE = """
Вы Роутер в Telegram боте для управления здоровьем и расписанием.
Определите, к какому агенту направить запрос пользователя.

Доступные агенты:
1. health_agent - Физическое здоровье (вода, упражнения, шаги, сон)
2. mental_health_agent - Ментальное здоровье (настроение, стресс, медитация, социальные контакты)
3. schedule_agent - Рабочее расписание (рабочие часы, перерывы, задачи, встречи)
4. summary_agent - Подведение итогов дня (достижения, вызовы, общий анализ)

Запрос пользователя: {user_request}
Текущее состояние: {user_state}

ВАЖНО: Ответ должен быть дружелюбным для Telegram чата.

Отвечайте в JSON формате:
{{
  "next_agent": "название_агента",
  "reasoning": "краткое обоснование",
  "message_to_agent": "сообщение для агента"
}}
"""

HEALTH_AGENT_SYSTEM_TEMPLATE = """
Вы Агент физического здоровья в Telegram боте.
Помогаете пользователю отслеживать физическое здоровье.

Доступные инструменты:
- log_water_intake: записать воду
- log_exercise: записать упражнения  
- log_steps: записать шаги
- log_sleep: записать сон

Текущие показатели:
💧 Вода: {water_intake}л
👣 Шаги: {steps_count}
🏃 Упражнения: {exercise_minutes} мин
😴 Сон: {sleep_hours} ч

Запрос: {user_request}

Цели дня: 2-3л воды, 10000+ шагов, 30+ мин упражнений, 7-9ч сна.

Будьте дружелюбны и мотивирующи! Используйте эмодзи в ответах.
"""

MENTAL_HEALTH_AGENT_SYSTEM_TEMPLATE = """
Вы Агент ментального здоровья в Telegram боте.
Поддерживаете психическое благополучие пользователя.

Доступные инструменты:
- update_mood: обновить настроение (1-10)
- update_stress: обновить стресс (1-10)
- log_meditation: записать медитацию
- log_social_interaction: записать общение

Текущее состояние:
😊 Настроение: {mood_score}/10
😰 Стресс: {stress_level}/10  
🧘 Медитация: {meditation_minutes} мин
👥 Общение: {social_interactions}

Запрос: {user_request}

Будьте эмпатичны и поддерживающи. Предлагайте конкретные техники.
Используйте эмодзи для создания теплой атмосферы.
"""

SCHEDULE_AGENT_SYSTEM_TEMPLATE = """
Вы Агент рабочего расписания в Telegram боте.
Помогаете с организацией рабочего времени.

Доступные инструменты:
- log_work_hours: записать рабочие часы
- log_break: записать перерыв
- complete_task: отметить задачу
- log_meeting: записать встречу

Текущие показатели:
⏰ Работа: {work_hours}ч
☕ Перерывы: {breaks_taken}
✅ Задачи: {tasks_completed}
🤝 Встречи: {meetings_attended}

Запрос: {user_request}

Помогайте планировать и поддерживать work-life баланс.
Используйте деловые эмодзи и будьте конструктивны.
"""

SUMMARY_AGENT_SYSTEM_TEMPLATE = """
Вы Агент подведения итогов в Telegram боте.
Анализируете день пользователя и помогаете с рефлексией.

Доступные инструменты:
- add_achievement: добавить достижение
- add_challenge: добавить вызов

Состояние пользователя: {full_user_state}
Последние активности: {daily_activities}
Достижения: {achievements}
Вызовы: {challenges}

Запрос: {user_request}

Будьте позитивны и мотивирующи. Выделяйте прогресс и рост.
Используйте эмодзи для создания вдохновляющей атмосферы.
"""


class HealthScheduleState(MessagesState):
    """Состояние для системы управления здоровьем и расписанием"""
    user_id: int
    user_request: str
    current_agent: str
    next_agent: str
    conversation_complete: bool
    last_agent_response: str
    routing_decision: str


# Агенты системы
def create_agents_for_user(user_id: int):
    """Создает агентов для конкретного пользователя"""
    user_state = get_user_state(user_id)
    tools_dict = create_user_tools(user_id)
    
    def router_agent(state: HealthScheduleState) -> Dict:
        logger.info(f"[Router] User {user_id}: Processing request")
        
        system_message = ROUTER_SYSTEM_TEMPLATE.format(
            user_request=state["user_request"],
            user_state=user_state.get_health_summary()
        )
        
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=state["user_request"])
        ])
        
        try:
            parsed_response = json.loads(response.content)
            next_agent = parsed_response.get("next_agent", "health_agent")
            reasoning = parsed_response.get("reasoning", "")
            message_to_agent = parsed_response.get("message_to_agent", "")
            
            logger.info(f"[Router] User {user_id}: Routing to {next_agent}")
            
            return {
                "messages": state["messages"] + [response],
                "next_agent": next_agent,
                "current_agent": "router",
                "routing_decision": reasoning,
                "last_agent_response": message_to_agent
            }
        except json.JSONDecodeError as e:
            logger.error(f"[Router] JSON parsing error: {e}")
            return {
                "messages": state["messages"] + [response],
                "next_agent": "health_agent",
                "current_agent": "router"
            }

    def health_agent(state: HealthScheduleState) -> Dict:
        logger.info(f"[Health Agent] User {user_id}: Processing health request")
        
        system_message = HEALTH_AGENT_SYSTEM_TEMPLATE.format(
            water_intake=user_state.water_intake,
            steps_count=user_state.steps_count,
            exercise_minutes=user_state.exercise_minutes,
            sleep_hours=user_state.sleep_hours,
            user_request=state["user_request"]
        )
        
        llm_with_tools = llm.bind_tools(tools_dict['health_tools'])
        response = llm_with_tools.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=state["user_request"])
        ])
        
        return {
            "messages": state["messages"] + [response],
            "current_agent": "health_agent",
            "next_agent": "complete",
            "last_agent_response": response.content
        }

    def mental_health_agent(state: HealthScheduleState) -> Dict:
        logger.info(f"[Mental Health Agent] User {user_id}: Processing mental health request")
        
        system_message = MENTAL_HEALTH_AGENT_SYSTEM_TEMPLATE.format(
            mood_score=user_state.mood_score,
            stress_level=user_state.stress_level,
            meditation_minutes=user_state.meditation_minutes,
            social_interactions=user_state.social_interactions,
            user_request=state["user_request"]
        )
        
        llm_with_tools = llm.bind_tools(tools_dict['mental_health_tools'])
        response = llm_with_tools.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=state["user_request"])
        ])
        
        return {
            "messages": state["messages"] + [response],
            "current_agent": "mental_health_agent",
            "next_agent": "complete",
            "last_agent_response": response.content
        }

    def schedule_agent(state: HealthScheduleState) -> Dict:
        logger.info(f"[Schedule Agent] User {user_id}: Processing schedule request")
        
        system_message = SCHEDULE_AGENT_SYSTEM_TEMPLATE.format(
            work_hours=user_state.work_hours,
            breaks_taken=user_state.breaks_taken,
            tasks_completed=len(user_state.tasks_completed),
            meetings_attended=user_state.meetings_attended,
            user_request=state["user_request"]
        )
        
        llm_with_tools = llm.bind_tools(tools_dict['schedule_tools'])
        response = llm_with_tools.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=state["user_request"])
        ])
        
        return {
            "messages": state["messages"] + [response],
            "current_agent": "schedule_agent", 
            "next_agent": "complete",
            "last_agent_response": response.content
        }

    def summary_agent(state: HealthScheduleState) -> Dict:
        logger.info(f"[Summary Agent] User {user_id}: Processing summary request")
        
        system_message = SUMMARY_AGENT_SYSTEM_TEMPLATE.format(
            full_user_state=user_state.get_health_summary(),
            daily_activities=user_state.daily_activities[-5:],
            achievements=user_state.achievements,
            challenges=user_state.challenges,
            user_request=state["user_request"]
        )
        
        llm_with_tools = llm.bind_tools(tools_dict['summary_tools'])
        response = llm_with_tools.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=state["user_request"])
        ])
        
        return {
            "messages": state["messages"] + [response],
            "current_agent": "summary_agent",
            "next_agent": "complete",
            "conversation_complete": True,
            "last_agent_response": response.content
        }

    return {
        'router': router_agent,
        'health_agent': health_agent,
        'mental_health_agent': mental_health_agent,
        'schedule_agent': schedule_agent,
        'summary_agent': summary_agent,
        'all_tools': sum(tools_dict.values(), [])
    }


def build_user_graph(user_id: int):
    """Создает граф для конкретного пользователя"""
    agents = create_agents_for_user(user_id)
    
    builder = StateGraph(HealthScheduleState)
    
    # Добавляем узлы
    builder.add_node("router", agents['router'])
    builder.add_node("health_agent", agents['health_agent'])
    builder.add_node("mental_health_agent", agents['mental_health_agent'])
    builder.add_node("schedule_agent", agents['schedule_agent'])
    builder.add_node("summary_agent", agents['summary_agent'])
    builder.add_node("tools", ToolNode(agents['all_tools']))
    
    # Стартовый узел
    builder.add_edge(START, "router")
    
    # Маршрутизация от роутера
    builder.add_conditional_edges(
        "router",
        lambda x: x["next_agent"],
        {
            "health_agent": "health_agent",
            "mental_health_agent": "mental_health_agent",
            "schedule_agent": "schedule_agent", 
            "summary_agent": "summary_agent"
        }
    )
    
    # Условные переходы для инструментов
    def should_use_tools(state):
        if not state["messages"]:
            return "no_tools"
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "use_tools"
        return "no_tools"
    
    for agent_name in ["health_agent", "mental_health_agent", "schedule_agent", "summary_agent"]:
        builder.add_conditional_edges(
            agent_name,
            should_use_tools,
            {"use_tools": "tools", "no_tools": END}
        )
    
    builder.add_edge("tools", END)
    
    return builder.compile()


async def process_user_request(user_id: int, user_request: str) -> str:
    """Обрабатывает запрос пользователя через систему агентов"""
    try:
        logger.info(f"Processing request from user {user_id}: {user_request}")
        
        # Создаем граф для пользователя
        graph = build_user_graph(user_id)
        
        # Начальное состояние
        initial_state = {
            "user_id": user_id,
            "messages": [],
            "user_request": user_request,
            "current_agent": "",
            "next_agent": "router",
            "conversation_complete": False,
            "last_agent_response": "",
            "routing_decision": ""
        }
        
        # Запускаем граф
        final_state = graph.invoke(initial_state, {"recursion_limit": 10})
        
        # Получаем последний ответ
        if final_state["messages"]:
            last_message = final_state["messages"][-1]
            response_text = last_message.content
            
            # Если есть вызовы инструментов, добавляем их результаты
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                response_text += "\n\n✅ Данные записаны!"
            
            return response_text
        else:
            return "Извините, произошла ошибка при обработке вашего запроса."
            
    except Exception as e:
        logger.error(f"Error processing request for user {user_id}: {e}")
        return f"Произошла ошибка: {str(e)}"


# Telegram Bot Handlers
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Обработчик команд /start и /help"""
    user_id = message.from_user.id
    
    # Создаем пользователя если его нет
    get_user_state(user_id)
    
    welcome_text = """
🌟 *Добро пожаловать в бота управления здоровьем!* 🌟

Я помогу вам отслеживать:
💪 Физическое здоровье (вода, упражнения, шаги, сон)
🧠 Ментальное состояние (настроение, стресс, медитация)
💼 Рабочее расписание (задачи, встречи, перерывы)
📊 Подведение итогов дня

*Примеры сообщений:*
• "Выпил 2 стакана воды"
• "Прошел 5000 шагов" 
• "Настроение 8 из 10"
• "Работал 4 часа"
• "Подведи итоги дня"

*Команды:*
/stats - показать статистику дня
/reset - сбросить данные дня
/help - показать эту справку

Просто пишите мне о своих активностях, и я все запишу! 📝
"""
    
    bot.reply_to(message, welcome_text, parse_mode='Markdown')


@bot.message_handler(commands=['stats'])
def send_stats(message):
    """Показать статистику пользователя"""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    
    stats_text = user_state.get_daily_summary_text()
    bot.reply_to(message, stats_text, parse_mode='Markdown')


@bot.message_handler(commands=['reset'])
def reset_user_data(message):
    """Сброс данных пользователя"""
    user_id = message.from_user.id
    
    # Создаем клавиатуру подтверждения
    markup = types.InlineKeyboardMarkup()
    markup.add(
        types.InlineKeyboardButton("✅ Да, сбросить", callback_data=f"reset_confirm_{user_id}"),
        types.InlineKeyboardButton("❌ Отмена", callback_data="reset_cancel")
    )
    
    bot.reply_to(
        message, 
        "⚠️ Вы уверены, что хотите сбросить все данные за сегодня?", 
        reply_markup=markup
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith("reset"))
def handle_reset_callback(call):
    """Обработчик подтверждения сброса данных"""
    if call.data.startswith("reset_confirm"):
        user_id = int(call.data.split("_")[-1])
        
        # Сбрасываем данные пользователя
        user_states[user_id] = UserHealthState(user_id=user_id)
        
        bot.edit_message_text(
            "✅ Данные успешно сброшены! Начните новый день с чистого листа 🌅",
            call.message.chat.id,
            call.message.message_id
        )
        logger.info(f"User {user_id} reset their data")
        
    elif call.data == "reset_cancel":
        bot.edit_message_text(
            "❌ Сброс отменен. Ваши данные сохранены.",
            call.message.chat.id,
            call.message.message_id
        )


@bot.message_handler(func=lambda message: True)
def handle_user_message(message):
    """Основной обработчик сообщений пользователей"""
    user_id = message.from_user.id
    user_request = message.text
    
    logger.info(f"Received message from user {user_id}: {user_request}")
    
    # Отправляем индикатор "печатаю"
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Используем синхронную версию для совместимости с telebot
        response = process_user_request_sync(user_id, user_request)
        
        # Отправляем ответ
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error handling message from user {user_id}: {e}")
        bot.reply_to(
            message, 
            "😕 Произошла ошибка при обработке вашего сообщения. Попробуйте еще раз или обратитесь к /help"
        )


def process_user_request_sync(user_id: int, user_request: str) -> str:
    """Синхронная версия обработки запроса пользователя"""
    try:
        logger.info(f"Processing request from user {user_id}: {user_request}")
        
        # Создаем граф для пользователя
        graph = build_user_graph(user_id)
        
        # Начальное состояние
        initial_state = {
            "user_id": user_id,
            "messages": [],
            "user_request": user_request,
            "current_agent": "",
            "next_agent": "router",
            "conversation_complete": False,
            "last_agent_response": "",
            "routing_decision": ""
        }
        
        # Запускаем граф
        final_state = graph.invoke(initial_state, {"recursion_limit": 10})
        
        # Получаем последний ответ
        if final_state["messages"]:
            last_message = final_state["messages"][-1]
            response_text = last_message.content
            
            # Если есть вызовы инструментов, добавляем подтверждение
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                response_text += "\n\n✅ *Данные успешно записаны!*"
            
            # Ограничиваем длину ответа для Telegram
            if len(response_text) > 4000:
                response_text = response_text[:3900] + "\n\n... (сообщение обрезано)"
            
            return response_text
        else:
            return "😕 Извините, произошла ошибка при обработке вашего запроса."
            
    except Exception as e:
        logger.error(f"Error processing request for user {user_id}: {e}")
        return f"❌ Произошла ошибка: {str(e)[:200]}..."


def create_daily_report_keyboard():
    """Создает клавиатуру для быстрых действий"""
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False)
    
    # Первый ряд - физическое здоровье
    markup.row("💧 Выпил воду", "👣 Прошел шаги")
    markup.row("🏃 Упражнения", "😴 Записать сон")
    
    # Второй ряд - ментальное здоровье  
    markup.row("😊 Настроение", "😰 Уровень стресса")
    markup.row("🧘 Медитация", "👥 Общение")
    
    # Третий ряд - работа
    markup.row("💼 Рабочие часы", "✅ Выполнил задачу")
    markup.row("☕ Перерыв", "🤝 Встреча")
    
    # Четвертый ряд - итоги и статистика
    markup.row("📊 Статистика", "🌟 Итоги дня")
    
    return markup


@bot.message_handler(commands=['keyboard'])
def show_keyboard(message):
    """Показать клавиатуру быстрых действий"""
    markup = create_daily_report_keyboard()
    bot.reply_to(
        message,
        "🎯 Используйте кнопки для быстрого ввода данных или пишите свободным текстом:",
        reply_markup=markup
    )


@bot.message_handler(commands=['hide'])
def hide_keyboard(message):
    """Скрыть клавиатуру"""
    markup = types.ReplyKeyboardRemove()
    bot.reply_to(message, "👍 Клавиатура скрыта", reply_markup=markup)


# Планировщик для отправки напоминаний
def schedule_daily_reminders():
    """Планирует ежедневные напоминания пользователям"""
    import schedule
    import time
    from threading import Thread
    
    def send_morning_reminder():
        """Утреннее напоминание"""
        for user_id in user_states.keys():
            try:
                reminder_text = """
🌅 *Доброе утро!* 

Начните день правильно:
💧 Выпейте стакан воды
🧘 5 минут медитации
📝 Поставьте цели на день

Напишите мне о своих утренних активностях!
"""
                bot.send_message(user_id, reminder_text, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending morning reminder to user {user_id}: {e}")
    
    def send_evening_reminder():
        """Вечернее напоминание"""
        for user_id in user_states.keys():
            try:
                reminder_text = """
🌆 *Время подвести итоги дня!*

Напишите мне:
• Что удалось сегодня? 
• Какие были вызовы?
• Как себя чувствуете?

Команда /stats покажет статистику дня 📊
"""
                bot.send_message(user_id, reminder_text, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending evening reminder to user {user_id}: {e}")
    
    def send_water_reminder():
        """Напоминание о воде"""
        for user_id, user_state in user_states.items():
            try:
                if user_state.water_intake < 1.5:  # Если выпил меньше 1.5 литров
                    bot.send_message(
                        user_id, 
                        f"💧 Напоминание: вы выпили {user_state.water_intake}л воды. Не забывайте пить!"
                    )
            except Exception as e:
                logger.error(f"Error sending water reminder to user {user_id}: {e}")
    
    # Планируем напоминания
    schedule.every().day.at("08:00").do(send_morning_reminder)
    schedule.every().day.at("20:00").do(send_evening_reminder) 
    schedule.every().hour.do(send_water_reminder)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    # Запускаем планировщик в отдельном потоке
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Daily reminders scheduler started")


def export_user_data(user_id: int) -> str:
    """Экспорт данных пользователя в JSON"""
    user_state = get_user_state(user_id)
    
    export_data = {
        "user_id": user_id,
        "export_date": datetime.now().isoformat(),
        "health_summary": user_state.get_health_summary(),
        "daily_activities": user_state.daily_activities,
        "achievements": user_state.achievements,
        "challenges": user_state.challenges,
        "tasks_completed": user_state.tasks_completed
    }
    
    return json.dumps(export_data, ensure_ascii=False, indent=2)


@bot.message_handler(commands=['export'])
def export_data(message):
    """Экспорт данных пользователя"""
    user_id = message.from_user.id
    
    try:
        export_json = export_user_data(user_id)
        
        # Создаем файл
        filename = f"health_data_{user_id}_{datetime.now().strftime('%Y%m%d')}.json"
        
        bot.send_document(
            message.chat.id,
            document=export_json.encode('utf-8'),
            visible_file_name=filename,
            caption="📊 Ваши данные за сегодня"
        )
        
    except Exception as e:
        logger.error(f"Error exporting data for user {user_id}: {e}")
        bot.reply_to(message, "❌ Ошибка при экспорте данных")


# Функция для получения статистики бота
def get_bot_statistics():
    """Получает общую статистику бота"""
    total_users = len(user_states)
    active_users_today = sum(1 for state in user_states.values() if state.daily_activities)
    total_activities = sum(len(state.daily_activities) for state in user_states.values())
    
    return {
        "total_users": total_users,
        "active_users_today": active_users_today, 
        "total_activities": total_activities
    }


@bot.message_handler(commands=['admin'])
def admin_stats(message):
    """Команда для администратора (показать статистику бота)"""
    # Здесь можно добавить проверку на права администратора
    admin_ids = [22286014]  # Замените на ID администраторов
    
    if message.from_user.id not in admin_ids:
        bot.reply_to(message, "❌ У вас нет прав администратора")
        return
    
    stats = get_bot_statistics()
    
    admin_text = f"""
🔧 *Статистика бота:*

👥 Всего пользователей: {stats['total_users']}
✅ Активных сегодня: {stats['active_users_today']}  
📊 Всего активностей: {stats['total_activities']}

Время работы: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    bot.reply_to(message, admin_text, parse_mode='Markdown')


def main():
    """Основная функция запуска бота"""
    logger.info("Starting Telegram Health Bot...")
    
    # Запускаем планировщик напоминаний
    schedule_daily_reminders()
    
    # Информация о запуске
    print("🤖 Telegram Health Bot запущен!")
    print("📱 Отправьте /start боту для начала работы")
    print("⚡ Нажмите Ctrl+C для остановки")
    
    # Запускаем бота
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n👋 Бот остановлен")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        print(f"❌ Ошибка бота: {e}")


if __name__ == "__main__":
    # Проверяем наличие токенов
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("❌ Установите TELEGRAM_BOT_TOKEN!")
        exit(1)
    
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("❌ Установите OPENAI_API_KEY!")
        exit(1)
    
    main()