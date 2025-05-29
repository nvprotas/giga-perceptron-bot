import logging
import json
import random
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import telebot
from telebot import types
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import re

from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GIGACHAT_API_KEY = os.getenv("GIGA_CHAT_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")
MODEL = os.getenv("MODEL")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Основные константы для скоров
PARAMS = [
    "сон", "активность", "питание", "чтение", "ментальное", "медитация", "спорт", "нагрузка по работе", "шаги"
]
CATS = ["отлично", "хорошо", "удовлетворительно", "плохо"]
YN = ["да", "нет"]

# ================================
#   Состояние пользователя
# ================================

@dataclass
class UserHealthState:
    user_id: int
    user_name: Optional[str] = None
    dialog_history: List[str] = field(default_factory=list)
    interaction_state: Optional[str] = None      # FSM: None|'goal'|'collect_data'|'confirm_generation'|'showing_history'|'daily_update'|'chat'
    health_goal: Optional[str] = None
    input_answers: Dict[str, Any] = field(default_factory=dict)     # анкета
    history_data: List[Dict[str, Any]] = field(default_factory=list)# 7 дней данных
    current_day: int = 0
    total_score: float = 0.0
    waiting_for_params: bool = False  # ждем ввод параметров от пользователя

    # Simple in-memory agent memory for context preservation
    _memory_context: str = ""

    def add_message(self, message: str, from_user: bool):
        tag = "👤" if from_user else "🤖"
        logger.debug(f"Add message to history | user_id={self.user_id} | from_user={from_user} | msg={message}")
        self.dialog_history.append(f"{tag} {message}")
        # Also update memory context with latest message
        self.update_memory(f"{tag} {message}")

    def reset_dialog(self):
        logger.info(f"Resetting dialog and state for user_id={self.user_id}")
        self.dialog_history = []
        self.interaction_state = None
        self.health_goal = None
        self.input_answers = {}
        self.history_data = []
        self.current_day = 0
        self.total_score = 0.0
        self.waiting_for_params = False
        self._memory_context = ""

    def get_context(self):
        # Return memory context if available, else dialog history last 8 messages
        if self._memory_context:
            return self._memory_context
        else:
            return "\n".join(self.dialog_history[-10:])

    def update_memory(self, new_text: str, max_length: int = 2000):
        """Update the memory context with new text, trimming if needed."""
        if not self._memory_context:
            self._memory_context = new_text
        else:
            self._memory_context = f"{self._memory_context}\n{new_text}"
        # Trim context if it gets too long
        if len(self._memory_context) > max_length:
            # Keep only the last max_length characters intelligently (by lines)
            lines = self._memory_context.splitlines()
            trimmed_lines = []
            total_len = 0
            for line in reversed(lines):
                total_len += len(line) + 1
                if total_len > max_length:
                    break
                trimmed_lines.append(line)
            trimmed_lines.reverse()
            self._memory_context = "\n".join(trimmed_lines)

    def generate_default_params(self):
        """Генерирует случайные параметры пользователя при старте"""
        genders = ["мужчина", "женщина"]
        activity_levels = ["низкий", "средний", "высокий"]
        stress_levels = ["низкий", "средний", "высокий"]

        self.input_answers = {
            "пол": random.choice(genders),
            "возраст": random.randint(20, 60),
            "рост": random.randint(155, 190),
            "вес": random.randint(55, 100),
            "активность": random.choice(activity_levels),
            "уровень стресса": random.choice(stress_levels),
            "курение": random.choice(YN),
            "алкоголь": random.choice(YN),
            "спорт": random.choice(YN),
            "чтение": random.choice(YN),
            "медитация": random.choice(YN)
        }
        logger.info(f"Default parameters generated for user_id={self.user_id}: {self.input_answers}")

# users state storage:
user_states: Dict[int, UserHealthState] = {}

def get_user(user_id: int, user_name: Optional[str] = None) -> UserHealthState:
    if user_id not in user_states:
        logger.info(f"New user session started: user_id={user_id}")
        user_states[user_id] = UserHealthState(user_id, user_name=user_name)
    else:
        # обновим имя, если оно появилось
        if user_name and not user_states[user_id].user_name:
            user_states[user_id].user_name = user_name
    return user_states[user_id]

# ================================
#   LLM и Prompt'ы
# ================================

if MODEL_PROVIDER == "openai":
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.6,
        api_key=OPENAI_API_KEY,
        max_completion_tokens=1024,
        streaming=False
    )
    print(llm.invoke("test"))
elif MODEL_PROVIDER == "gigachat":
    from langchain_gigachat.chat_models import GigaChat
    print(GIGACHAT_API_KEY)
    llm = GigaChat(
        model=MODEL,
        credentials=GIGACHAT_API_KEY,
        temperature=0.6,
        max_completion_tokens=2048,
        streaming=False,
        verify_ssl_certs=False,
        # scope="GIGACHAT_API_CORP"
        scope="GIGACHAT_API_PERS"
    )
    print(llm.invoke("test"))
else:
    raise ValueError(f"Unsupported model provider: {MODEL_PROVIDER}. Supported: 'openai', 'gigachat'.")

SYSTEM_ASK_GOAL = (
    """
    Ты ассистент-коуч по здоровью. Не здоровайся с пользователем.
    Спроси у пользователя, к какой цели он хочет прийти в работе с ботом: например, похудеть, набрать массу, улучшить сон и т.д. 
    Не предлагай варианты, спроси свободно.
    """
)
SYSTEM_ASK_FORM = (
    """
    Ты умный ассистент по здоровью. Сообщи пользователю, что для него автоматически были загружены базовые параметры и покажи их. 
    Предложи начать работу с этими параметрами. Уточни, верно ли загружены параметры?
    """
)
SYSTEM_REPORT = (
    """
    Используя описанные ниже исторические данные пользователя (7 дней), 
    создай для него краткий отчёт в стиле коуча-бота: что было лучше, что хуже, на что стоит обратить внимание. 
    Добавь мотивационные советы. 
    Используй эмодзи. Баллы в выдаче не комментируй, выводи прогресс как достижение цели.
    """
)
SYSTEM_DAILY_REPORT = (
    """
    На основе сегодняшних данных скора и параметров создай короткий отчет-поддержку для пользователя (максимум 4 предложения). 
    Если заметен прогресс, упомяни это. Заверши мотивацией и предложи продолжать на пути к цели."""
)
SYSTEM_CHAT = (
    "Ты ассистент-коуч по здоровью. Отвечай на вопросы пользователя, давай советы по здоровью, мотивируй. "
    "Учитывай контекст диалога и цель пользователя. Будь дружелюбным и поддерживающим. "
)

# ================================
#   Вспомогательные функции
# ================================

def random_day_params(user_info: dict) -> Dict[str, Any]:
    """
    Генерирует параметры дня — приближенно на основе данных анкеты пользователя
    """
    # Для "лучших" исходных — больше excellent/good, худшие — хуже.
    base = 3 if (user_info.get('активность', 'средний') in ['низкий']) else 4
    activity_bias = base + (1 if user_info.get('спорт', 'нет') == 'да' else 0)
    stress_magic = 1 if user_info.get('уровень стресса', '').lower() == 'высокий' else 0

    # Генерация случайного количества шагов (например, от 2000 до 15000)
    steps = random.randint(2000, 15000)

    params = {
        "дата": "",
        "сон": random.choices(CATS, [0.1, 0.30, 0.30, 0.30])[0],
        "активность": random.choices(CATS, [0.10, 0.30, 0.3, 0.3])[0] if activity_bias > 3 else random.choices(CATS, [0.2, 0.3, 0.3, 0.2])[0],
        "питание": random.choices(CATS, [0.15, 0.40, 0.30, 0.15])[0],
        "чтение": random.choice(YN),
        "ментальное": random.choices(CATS, [0.15, 0.35, 0.35, 0.15])[0] if not stress_magic else random.choices(CATS, [0,0.20,0.40,0.40])[0],
        "медитация": random.choice(YN),
        "спорт": random.choice(YN if activity_bias > 3 else ["нет", "нет", "да", "нет"]),
        "нагрузка по работе": random.choices(CATS, [0.25, 0.4, 0.25, 0.10])[0],
        "шаги": steps
    }
    logger.debug(f"Random day params generated: {params}")
    return params

def params_to_score(params: Dict[str,Any]) -> float:
    """
    Выставляет дробный скор за 1 день — max 5.5 баллов (с учетом шагов)
    """
    score = 0.0
    mapping = {"отлично": 1.0, "хорошо": 0.75, "удовлетворительно": 0.45, "плохо": 0.10}
    # Сон, активность, питание, ментальное, нагрузка по работе: 0...1 за каждый
    for key in ["сон", "активность", "питание", "ментальное", "нагрузка по работе"]:
        score += mapping.get(params[key], 0.10)
    # за спорт и медитацию, чтение — +0.25 каждая если были (да)
    for k in ["медитация", "спорт", "чтение"]:
        score += 0.25 if params.get(k,"нет")=="да" else 0.0
    # шаги: если больше 10000 — +0.5, если 7000-9999 — +0.3, если 4000-6999 — +0.1, иначе 0
    steps = params.get("шаги", 0)
    if isinstance(steps, str):
        try:
            steps = int(steps.replace(" ", ""))
        except Exception:
            steps = 0
    if steps >= 10000:
        score += 0.5
    elif steps >= 7000:
        score += 0.3
    elif steps >= 4000:
        score += 0.1
    return min(score, 5.5)

def humanify_params(params):
    # Красивое форматирование для дневного отчёта с эмодзи для категорий
    order = [
        "сон", "активность", "питание", "чтение", "ментальное", "медитация", "спорт", "нагрузка по работе", "шаги"
    ]
    name_map = {
        "сон": "Сон",
        "активность": "Активность",
        "питание": "Питание",
        "чтение": "Чтение",
        "ментальное": "Ментальное",
        "медитация": "Медитация",
        "спорт": "Спорт",
        "нагрузка по работе": "Нагрузка по работе",
        "шаги": "Шаги"
    }
    emoji_map = {
        "отлично": "🟢",
        "хорошо": "🟡",
        "удовлетворительно": "🟠",
        "плохо": "🔴",
        "да": "✅",
        "нет": "❌"
    }
    lines = []
    for k in order:
        if k in params:
            v = params[k]
            if k == "шаги":
                vv = f"{v} 🚶"
            elif v in emoji_map:
                vv = f"{emoji_map[v]} {v.capitalize() if v not in ['да','нет'] else ''}".strip()
            else:
                vv = v
            lines.append(f"{name_map[k]}: {vv}")
    return "; ".join(lines)

def score_progress_bar(score, maxv=25):
    filled = int(score / maxv * 20)
    return "🏁 " + "█"*filled + "-"*(20-filled) + f" {score:.1f}/{maxv}"

def make_7days_history(user_info):
    """
    Генерирует историю на 7 дней:
    в каждом дне хранится только скор за этот день.
    """
    d0 = datetime.now() - timedelta(days=6)
    out = []
    for i in range(7):
        p = random_day_params(user_info)
        p['дата'] = (d0 + timedelta(days=i)).strftime("%d.%m")
        # ежедневный скор
        daily_score = round(params_to_score(p), 2)
        p['скор'] = daily_score
        out.append(p)
    logger.info(f"7 days history generated for params: {user_info}")
    logger.debug(f"History: {out}")
    return out


def next_day(user_state: UserHealthState):
    ui = user_state.input_answers
    day_params = random_day_params(ui)
    day_params['дата'] = (datetime.now()+timedelta(days=user_state.current_day)).strftime("%d.%m")
    day_params['скор'] = round(params_to_score(day_params),2)
    logger.info(f"Next simulated day generated for user_id={user_state.user_id}: {day_params}")
    return day_params

def format_user_params(user_info: dict) -> str:
    """Форматирует параметры пользователя для отображения"""
    return (
        f"👤 Ваши параметры:\n"
        f"• Пол: {user_info.get('пол', 'не указан')}\n"
        f"• Возраст: {user_info.get('возраст', 'не указан')} лет\n"
        f"• Рост: {user_info.get('рост', 'не указан')} см\n"
        f"• Вес: {user_info.get('вес', 'не указан')} кг\n"
        f"• Активность: {user_info.get('активность', 'не указана')}\n"
        f"• Уровень стресса: {user_info.get('уровень стресса', 'не указан')}\n"
        f"• Курение: {user_info.get('курение', 'не указано')}\n"
        f"• Алкоголь: {user_info.get('алкоголь', 'не указано')}\n"
        f"• Спорт: {user_info.get('спорт', 'не указано')}\n"
        f"• Чтение: {user_info.get('чтение', 'не указано')}\n"
        f"• Медитация: {user_info.get('медитация', 'не указано')}\n"
        f"• Шаги: {user_info.get('шаги', 'не указано')}"
    )

# FSM Dictionary (by user_id) - ключ состояния в анкете
daily_form_fields = [
    ("пол", "Укажите, пожалуйста, ваш пол (мужчина/женщина)"),
    ("возраст", "Ваш возраст (число лет):"),
    ("рост", "Ваш рост в сантиметрах:"),
    ("вес", "Ваш вес (в кг):"),
    ("активность", "Опишите ваш уровень ежедневной активности: низкий, средний или высокий?"),
    ("уровень стресса", "Какой у вас уровень стресса обычно: низкий, средний или высокий?"),
    ("курение", "Курите ли вы? (да/нет)"),
    ("алкоголь", "Употребляете ли вы алкоголь? (да/нет)"),
    ("шаги", "Сколько шагов вы проходите в среднем за день? (число)")
]

# ================================
#   LLM-помощник (вопросы и отчёты)
# ================================
def call_llm(messages: List[HumanMessage|SystemMessage]) -> str:
    try:
        logger.info("Calling LLM for response")
        logger.debug(f"LLM Messages: {messages}")
        response = llm.invoke(messages)
        logger.info(f"LLM response received: {getattr(response, 'content', None)}")
        logger.debug(f"LLM raw response: {getattr(response, 'content', None)}")
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM Call error: {e}")
        return "🤖 Ошибка при генерации ответа. Попробуйте ещё раз."

def ask_goal_message(user_state:UserHealthState):
    context = user_state.get_context()
    return call_llm([SystemMessage(SYSTEM_ASK_GOAL), HumanMessage(context)])

def ask_form_message(user_state:UserHealthState):
    params_text = format_user_params(user_state.input_answers)
    name = user_state.user_name or "пользователь"
    return (
        f"{params_text}\n\n"
        f"{name}, проверьте, пожалуйста, корректны ли эти параметры. Если всё верно — напишите 'всё ок' или 'да'. "
        "Если что-то не так — напишите, что нужно исправить."
    )

def report_history_message(user_state:UserHealthState):
    # Сделать компактный json историю пользователя
    user_days = user_state.history_data
    summary = "\n".join([
        f"{d['дата']}: {humanify_params(d)} (скор: {d['скор']})" for d in user_days
    ])
    system = (
        SYSTEM_REPORT
        + "\nОграничь длину анализа 900-1200 символами. Не превышай этот лимит.\n"
        + f"\n---\nИстория:\n{summary}\n---\nЦель пользователя: {user_state.health_goal or ''}\n"
    )
    return call_llm([SystemMessage(system)])

def day_report_message(user_state:UserHealthState, day_dict:dict):
    short_data = ", ".join([f"{k}:{v}" for k,v in day_dict.items() if k!="дата"])
    context = f"Сегодня: {short_data}\nОбщий балл: {user_state.total_score:.1f} из 25"
    system = SYSTEM_DAILY_REPORT
    return call_llm([SystemMessage(system), HumanMessage(context)])

def chat_response(user_state: UserHealthState, user_message: str):
    context = user_state.get_context()
    goal_info = f"Цель пользователя: {user_state.health_goal or 'не указана'}"
    progress_info = f"Текущий прогресс: день {user_state.current_day}, баллы {user_state.total_score:.1f}"
    system_msg = f"{SYSTEM_CHAT}\n\n{goal_info}\n{progress_info}"
    messages = [SystemMessage(system_msg), HumanMessage(f"Контекст диалога:\n{context}\n\nПоследнее сообщение пользователя: {user_message}")]
    response = call_llm(messages)
    logger.info(f"LLM response: {response}")
    return response

def next_button_markup():
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Следующий день", callback_data="next_sim_day"))
    return markup

def params_choice_markup():
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("🎲 Сгенерировать параметры", callback_data="generate_params"))
    markup.add(types.InlineKeyboardButton("✏️ Ввести параметры самому", callback_data="input_params"))
    return markup

def main_menu_markup():
    # Кнопка "Следующий день" будет показываться только после просмотра отчёта за день
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Следующий день", callback_data="next_sim_day"))
    return markup

def main_menu_keyboard():
    # Клавиатура больше не используется
    return None

# start_simulation_markup больше не нужен, так как кнопка не используется

def detect_user_intent(user_goal: str) -> str:
    """
    Обращается к LLM для детекции типа намерения пользователя.
    Возвращает один из вариантов: 'похудеть', 'набрать массу', 'улучшить сон', 'другое'
    """
    prompt = (
        "Пользователь назвал свою цель в работе над здоровьем. "
        "Определи, к какой из категорий она относится: "
        "1) коррекция веса (снизить вес, сбросить кг и т.п.), "
        "2) набрать массу, "
        "3) улучшить сон, "
        "4) другое. "
        "Только одна категория, без лишних символов, выведи ответ одним словом из этого списка: ['коррекция веса', 'набрать массу', 'улучшить сон', 'другое']"
        f"\n\nЦель пользователя: {user_goal}"
    )
    response = call_llm([SystemMessage(prompt)])
    return response.lower().strip()

# ================================
#         BOT HANDLERS FSM
# ================================

@bot.message_handler(commands=['start'])
def start_handler(message):
    # Получаем имя пользователя из Telegram
    tg_name = message.from_user.first_name or message.from_user.username or "пользователь"
    user = get_user(message.from_user.id, user_name=tg_name)
    logger.info(f"/start command invoked by user_id={message.from_user.id}")
    user.reset_dialog()
    user.user_name = tg_name
    user.interaction_state = 'goal'
    logger.info(f"State updated: user_id={user.user_id}, interaction_state='goal'")
    welcome = f"👋 Добро пожаловать, {tg_name}! Я ваш цифровой помощник для контроля и улучшения здоровья.\n\n"
    question = ask_goal_message(user)
    bot.send_message(message.chat.id, f"{welcome}{question}")
    user.add_message(question, from_user=False)

@bot.message_handler(commands=['help'])
def help_handler(message):
    user = get_user(message.from_user.id)
    logger.info(f"/help command invoked by user_id={message.from_user.id}")
    user.add_message(message.text, from_user=True)
    bot.reply_to(message, (
        "🤖 Я помогу вести путь к вашей цели (похудение, улучшение самочувствия и т.д.). "
        "Для старта отправьте /start. Вся логика поддерживает кнопки и диалог."
    ))

@bot.message_handler(func=lambda m: True)
def handle_all(message):
    # Обработка нажатия на кнопку "Следующий день" с клавиатуры
    if message.text.strip().lower() == "следующий день":
        # Эмулируем callback для next_sim_day
        fake_call = type('FakeCall', (), {})()
        fake_call.message = message
        fake_call.from_user = message.from_user
        next_sim_day_callback(fake_call)
        return
    user = get_user(message.from_user.id)
    text = message.text.strip()
    logger.info(f"Received message from user_id={user.user_id}: {text}")
    user.add_message(text, from_user=True)

    # FSM — этап выбора цели
    if user.interaction_state == "goal":
        user.health_goal = text
        logger.info(f"User goal set: user_id={user.user_id} | goal={text}")
        intent = detect_user_intent(text)
        logger.info(f"User goal intent resolved: {intent}")

        if intent == "коррекция веса":
            # Генерируем параметры автоматически
            user.generate_default_params()
            # Определяем пол по имени пользователя с помощью LLM
            if user.user_name:
                name = user.user_name.strip()
                gender_prompt = (
                    f"Определи пол по имени: '{name}'. "
                    "Ответь только одним словом: 'мужчина' или 'женщина'."
                )
                gender_response = call_llm([SystemMessage(gender_prompt)])
                gender = gender_response.lower().strip()
                if "жен" in gender:
                    user.input_answers["пол"] = "женщина"
                elif "муж" in gender:
                    user.input_answers["пол"] = "мужчина"
                else:
                    user.input_answers["пол"] = gender  # fallback, если LLM вернул что-то другое
                logger.info(f"Определён пол по имени '{name}': {user.input_answers['пол']}")
            user.interaction_state = 'collect_data'
            logger.info(f"State updated: user_id={user.user_id}, interaction_state='collect_data'")
            ask = ask_form_message(user)
            bot.send_message(message.chat.id, ask)
            user.add_message(ask, from_user=False)
        else:
            # Общаемся с пользователем через LLM без стандартной отбивки
            user.interaction_state = "chat"
            response = chat_response(user, text)
            bot.send_message(message.chat.id, response, reply_markup=main_menu_keyboard())
            user.add_message(response, from_user=False)
        return

    # FSM — этап подтверждения корректности параметров и генерации истории
    if user.interaction_state == "collect_data":
        logger.info(f"Collecting data from user_id={user.user_id} during 'collect_data' FSM stage")
        # Анализируем ответ пользователя с помощью LLM: подтверждает ли он корректность параметров?
        params_text = format_user_params(user.input_answers)
        prompt = (
            "Вот текущие параметры пользователя:\n"
            f"{params_text}\n"
            f"Пользователь написал: {text}\n"
            "Если пользователь подтвердил, что параметры корректны, верни 'подтверждено'. "
            "Если пользователь указал, что нужно что-то изменить, верни только изменённые параметры в формате ключ=значение, через запятую. "
            "Если изменений нет, верни 'нет изменений'."
        )
        llm_response = call_llm([SystemMessage(prompt)])
        logger.info(f"LLM param/confirmation response: {llm_response}")
        if "подтверждено" in llm_response.lower():
            # Генерируем историю 7 дней
            form_info = user.input_answers.copy()
            hist = make_7days_history(form_info)
            user.history_data = hist
            user.total_score = float(sum(d['скор'] for d in hist))
            user.current_day = 7
            user.interaction_state = "showing_history"
            logger.info(f"History for confirmed params generated for user_id={user.user_id}")
            report = report_history_message(user)
            bar = score_progress_bar(user.total_score)
            # Генерация индивидуального плана похудения на основе истории
            plan_prompt = (
                "На основе следующих параметров пользователя и его истории за 7 дней создай индивидуальную программу коррекции веса. "
                "План должен быть чётким, подробным и расписанным по дням (7 дней) по всем аспектам: спорт, питание, режим сна, ментальное здоровье, чтение, медитация и т.д. "
                "Для каждого дня укажи конкретные рекомендации по каждому аспекту. "
                "Форматируй план по дням: День 1: ... День 2: ... и т.д. "
                "Сделай текст мотивирующим и поддерживающим, но избегай воды. "
                "###Длина плана должна быть от 500 до 1000 символов. Не превышай этот лимит.### "
                "История пользователя:\n"
            )
            history_text = "\n".join([
                f"{d['дата']}: {humanify_params(d)} (скор: {d['скор']})" for d in hist
            ])
            params_text = format_user_params(user.input_answers)
            plan_message = call_llm([
                SystemMessage(plan_prompt),
                HumanMessage(f"{params_text}\n\n{history_text}")
            ])
            bot.send_message(
                message.chat.id,
                f"{report}\n\n{bar}\n\n📝 Ваш индивидуальный план похудения:\n\n{plan_message}",
                reply_markup=main_menu_keyboard()
            )
            user.add_message(report, from_user=False)
            user.add_message(plan_message, from_user=False)
        elif "нет изменений" in llm_response.lower():
            # Если LLM считает, что изменений нет, повторно просим подтвердить
            bot.send_message(message.chat.id, "Если всё верно, подтвердите это, либо напишите, что нужно изменить.")
        else:
            # Парсим ответ LLM и обновляем параметры
            try:
                pairs = llm_response.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        user.input_answers[key.strip()] = value.strip()
                # Показываем обновлённые параметры и снова просим подтвердить
                params_text = format_user_params(user.input_answers)
                bot.send_message(
                    message.chat.id,
                    f"Обновлённые параметры:\n{params_text}\n\nЕсли всё верно, подтвердите это, либо напишите, что нужно изменить."
                )
            except Exception as e:
                logger.error(f"Error parsing LLM param correction: {e}")
                bot.send_message(message.chat.id, "Не удалось распознать изменения. Пожалуйста, напишите, что нужно изменить, или подтвердите корректность параметров.")
        return

    # FSM — ввод параметров пользователем
    if user.waiting_for_params:
        try:
            logger.info(f"User entered custom params for day history: user_id={user.user_id}")
            # Парсим ввод пользователя (ожидаем формат: сон=хорошо, активность=отлично, и т.д.)
            params = {}
            pairs = text.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    params[key.strip()] = value.strip()

            # Генерируем историю с пользовательскими параметрами
            form_info = user.input_answers.copy()
            form_info.update(params)
            hist = make_7days_history(form_info)
            user.history_data = hist
            user.total_score = float(sum(d['скор'] for d in hist))
            user.current_day = 7
            user.interaction_state = "showing_history"
            user.waiting_for_params = False
            logger.info(f"History for entered params generated for user_id={user.user_id}")
            report = report_history_message(user)
            bar = score_progress_bar(user.total_score)
            bot.send_message(message.chat.id, f"{report}\n\n{bar}", reply_markup=main_menu_markup())
            user.add_message(report, from_user=False)
        except Exception as e:
            logger.error(f"Error parsing or generating history for user_id={user.user_id}: {e}")
            bot.send_message(message.chat.id, "Ошибка в формате. Попробуйте еще раз в формате: сон=хорошо, активность=отлично")
        return

    # FSM — режим чата
    if user.interaction_state == "chat":
        logger.info(f"User entered message in chat mode: user_id={user.user_id}")
        # Если пользователь в чате вдруг укажет цель похудеть, запускаем программу похудения
        intent = detect_user_intent(text)
        if intent == "коррекция веса":
            user.generate_default_params()
            user.interaction_state = 'collect_data'
            logger.info(f"User switched to weight correction program: user_id={user.user_id}")
            ask = ask_form_message(user)
            bot.send_message(message.chat.id, ask)
            user.add_message(ask, from_user=False)
            return
        response = chat_response(user, text)
        bot.send_message(message.chat.id, response, reply_markup=main_menu_keyboard())
        user.add_message(response, from_user=False)
        return

    # FSM — показываем историю, ждем действия пользователя (теперь текстом)
    if user.interaction_state == "showing_history":
        logger.info(f"User interacted during showing_history: user_id={user.user_id}")
        # Проверяем, хочет ли пользователь посмотреть отчет за текущий день
        report_prompt = (
            "Пользователь пишет:\n"
            f"'{text}'\n"
            "Если пользователь хочет посмотреть отчет за текущий день, верни 'отчет'. "
            "Если нет — верни 'нет'."
        )
        llm_response = call_llm([SystemMessage(report_prompt)])
        logger.debug(f"LLM response for day report intent: {llm_response}")
        if "отчет" in llm_response.lower():
            # Показываем статистику и отчет за день, а затем кнопку "Следующий день"
            day_dict = user.history_data[-1] if user.history_data else None
            if day_dict:
                day_text = day_report_message(user, day_dict)
                bar = score_progress_bar(user.total_score)
                report = (
                    f"📅 <b>{day_dict['дата']}</b>\n"
                    f"{humanify_params(day_dict)}\n"
                    f"<b>Скор:</b> <code>{day_dict['скор']:.2f}</code>\n"
                    f"<b>Сегодня:</b> <code>{day_dict['скор']:.2f}</code> баллов\n\n"
                    f"{day_text}\n\n{bar}"
                )
                bot.send_message(
                    message.chat.id,
                    report,
                    reply_markup=main_menu_markup(),
                    parse_mode='HTML'
                )
                user.interaction_state = "wait_next_day"
                return
            else:
                bot.send_message(message.chat.id, "Нет данных для отчёта за день.")
                return
        else:
            response = chat_response(user, text)
            bot.send_message(message.chat.id, response)
            user.add_message(response, from_user=False)
            return

    # FSM — ожидание перехода к следующему дню (только кнопка)
    if user.interaction_state == "wait_next_day":
        # Ждём нажатия на кнопку "Следующий день"
        return

    # FSM — обработка новых дней после старта 7-дневки
    if user.interaction_state == "daily_update":
        logger.info(f"User interacted during daily_update: user_id={user.user_id}")
        # Проверяем, хочет ли пользователь посмотреть отчет за текущий день
        report_prompt = (
            "Пользователь пишет:\n"
            f"'{text}'\n"
            "Если пользователь хочет посмотреть отчет за текущий день, верни 'отчет'. "
            "Если нет — верни 'нет'."
        )
        llm_response = call_llm([SystemMessage(report_prompt)])
        logger.debug(f"LLM response for day report intent: {llm_response}")
        if "отчет" in llm_response.lower():
            day_dict = user.history_data[-1] if user.history_data else None
            if day_dict:
                day_text = day_report_message(user, day_dict)
                bar = score_progress_bar(user.total_score)
                report = (
                    f"📅 <b>{day_dict['дата']}</b>\n"
                    f"{humanify_params(day_dict)}\n"
                    f"<b>Скор:</b> <code>{day_dict['скор']:.2f}</code>\n"
                    f"<b>Сегодня:</b> <code>{day_dict['скор']:.2f}</code> баллов\n\n"
                    f"{day_text}\n\n{bar}"
                )
                bot.send_message(
                    message.chat.id,
                    report,
                    reply_markup=main_menu_markup(),
                    parse_mode='HTML'
                )
                user.interaction_state = "wait_next_day"
                return
            else:
                bot.send_message(message.chat.id, "Нет данных для отчёта за день.")
                return
        else:
            response = chat_response(user, text)
            bot.send_message(message.chat.id, response)
            user.add_message(response, from_user=False)
            return

    # --- Если не FSM — запускать основного Health-LLM или отвечать дефолтом
    logger.info(f"No valid FSM state matched for user_id={user.user_id}. Sent default reply.")
    bot.reply_to(message, "👀 Пожалуйста, используйте /start для новой сессии похудения.\n(В этой демо-версии реализован только путь похудения с анализом по заданной анкете.)")

# Удалён обработчик callback start_simulation, так как кнопка больше не используется

@bot.callback_query_handler(func=lambda call: call.data == "generate_params")
def generate_params_callback(call):
    user = get_user(call.from_user.id)
    logger.info(f"User {user.user_id} requested generate_params callback")
    bot.edit_message_text("📂 Загружаю историю за последние 7 дней...",
                         chat_id=call.message.chat.id,
                         message_id=call.message.message_id)

    # Генерация истории 7 дней с ограничением по суммарному скору
    form_info = user.input_answers.copy()
    hist = make_7days_history(form_info)
    total_score = float(sum(d['скор'] for d in hist))
    attempts = 0
    max_attempts = 10
    while total_score > 25 and attempts < max_attempts:
        hist = make_7days_history(form_info)
        total_score = float(sum(d['скор'] for d in hist))
        attempts += 1
    if total_score > 25:
        total_score = 24.0
    user.history_data = hist
    user.total_score = total_score
    user.current_day = 7
    user.interaction_state = "showing_history"
    logger.info(f"History generated and state updated for user_id={user.user_id} (generate_params)")
    report = report_history_message(user)
    bar = score_progress_bar(user.total_score)
    bot.edit_message_text(f"{report}\n\n{bar}",
                         chat_id=call.message.chat.id,
                         message_id=call.message.message_id,
                         reply_markup=main_menu_markup())
    user.add_message(report, from_user=False)

    # Show generated default parameters to user
    default_params_str = "\n".join(f"{k}: {v}" for k, v in user.input_answers.items())
    bot.send_message(
        chat_id=call.message.chat.id,
        text=f"Загруженные параметры пользователя:\n{default_params_str}",
    )

@bot.callback_query_handler(func=lambda call: call.data == "input_params")
def input_params_callback(call):
    user = get_user(call.from_user.id)
    user.waiting_for_params = True
    logger.info(f"User_id={user.user_id} requested to input_params (switch to waiting_for_params=True)")
    params_text = (
        "Введите ваши параметры в формате:\n"
        "сон=хорошо, активность=отлично, питание=удовлетворительно, "
        "медитация=да, спорт=нет, чтение=да, шаги=8000\n\n"
        "Доступные значения:\n"
        "• Для сна, активности, питания, ментального состояния: отлично, хорошо, удовлетворительно, плохо\n"
        "• Для медитации, спорта, чтения, лекарств: да, нет\n"
        "• Для шагов: любое число (например, 8000)"
    )

    bot.edit_message_text(params_text,
                         chat_id=call.message.chat.id,
                         message_id=call.message.message_id)

@bot.callback_query_handler(func=lambda call: call.data == "start_chat")
def start_chat_callback(call):
    user = get_user(call.from_user.id)
    user.interaction_state = "chat"
    logger.info(f"User_id={user.user_id} switched to chat mode (interaction_state='chat')")
    bot.edit_message_text("💬 Теперь вы в режиме чата! Задавайте любые вопросы о здоровье, питании, тренировках. Я помогу вам советами и мотивацией!",
                         chat_id=call.message.chat.id,
                         message_id=call.message.message_id,
                         reply_markup=main_menu_markup())

@bot.callback_query_handler(func=lambda call: call.data == "next_sim_day")
def next_sim_day_callback(call):
    user = get_user(call.from_user.id)
    # Шаг 1: генерируем новый день
    day_dict = next_day(user)
    user.history_data.append(day_dict)
    user.current_day += 1

    # Шаг 2: пересчитываем общий скор как сумму скорoв последних 7 дней
    last_seven = user.history_data[-7:]
    user.total_score = sum(d['скор'] for d in last_seven)

    # После генерации дня переводим в состояние ожидания текстового запроса на отчет
    user.interaction_state = "daily_update"
    bot.send_message(
        call.message.chat.id,
        "Начинается новый день!🌞\nНе забудь посмотреть результаты в конце дня.",
    )

    # Конгратуляции, если порог превышен
    if user.total_score > 25:
        bot.send_message(call.message.chat.id, "🎉 Поздравляем! Ваш общий счет превысил пороговое значение 25!")

    # Завершение симуляции через 14 дней
    if user.current_day >= 21:
        bot.send_message(call.message.chat.id, "🎉 Вы прошли 21 день! Чтобы начать заново, отправьте /start")
        user.reset_dialog()


# ================================
#   Запуск
# ================================

def main():
    print("🤖 Telegram Health Coach Bot: стартуем!")
    logger.info("Bot polling started")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)

if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN :
        print("❌ Требуются TELEGRAM_BOT_TOKEN в .env")
    else:
        main()
