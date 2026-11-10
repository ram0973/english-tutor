"""
Голосовой репетитор английского «Джон из Чикаго»: hands-free + Silero VAD + перебивание + ПАМЯТЬ.
Микрофон слушается постоянно: Silero VAD сам находит начало и конец вашей фразы,
дальше Whisper -> Ollama -> озвучка (Piper/Kokoro). Если начать говорить, пока репетитор
отвечает, ответ обрывается (генерация и озвучка), и он слушает вас.

В окне каждая английская фраза - и ваша, и Джона - идёт вместе с переводом на русский
в скобках чёрным курсивом. Английский текст цветной, перевод - чёрный.

ВАЖНО: перебивание по умолчанию ВЫКЛЮЧЕНО. Оно нормально работает в НАУШНИКАХ - включите
галочку «Перебивание». С колонками микрофон слышит голос репетитора и принимает его за вас,
поэтому перебивание лучше не включать: пока репетитор думает и говорит, микрофон будет
игнорироваться (прервать можно кнопкой «Прервать»).

Запуск:  python english_tutor_memory.py
Нужно заранее: Python 3.10+ и установленная Ollama (https://ollama.com/download).
Остальное (pip-пакеты, ollama serve, модель, голос) скрипт делает сам.
"""
import collections
import json
import importlib.util
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import warnings
from datetime import date, datetime, timedelta
from tkinter import messagebox, ttk

# Silero VAD грузит свою модель через torch.jit.load, а torch 2.x ругается на это
# FutureWarning-ом. Предупреждение безобидное и всплывает в консоли при каждом запуске - гасим.
warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.*")

# ---------------------------- НАСТРОЙКИ ----------------------------
OLLAMA_MODEL = "gemma3:12b"       # ~8 ГБ VRAM; любая другая модель Ollama
#OLLAMA_MODEL = "nemotron-3.5-bolt:e4b"
# Замеры на RTX 5060 Ti 16 ГБ с этим промптом (num_ctx 8192, ответ «фраза + перевод + Fix»):
#   gemma3:12b  46 ток/с, первая фраза через 0.6 с, весь ответ ~2.5 с, ошибки находит 5/5
#   gemma3:4b  108 ток/с, первая фраза через 0.6 с, весь ответ ~0.8 с, ошибки находит 2/5
#   qwen3:8b    70 ток/с, но путает формат «фраза + перевод в скобках» (и думает перед ответом)
#   qwen2.5:7b  62 ток/с, тоже не держит формат
# Хотите совсем быстро и не важны исправления - поставьте "gemma3:4b".
WHISPER_MODEL = "large-v3-turbo"  # при проблемах с CUDA автоматически -> small на CPU
WHISPER_LANGUAGE = "en"  # язык распознавания; None - автоопределение (но тогда акцентный английский легко принимается за русский)
KEEP_ALIVE = "30m"
OPTIONS = {"temperature": 0.7, "num_ctx": 8192}  # 8192: в контекст помещается память о прошлых занятиях
SAMPLE_RATE = 16000
HISTORY_TURNS = 16

# --- личности репетитора: имя, возраст, город и голос ---
PERSONAS = {
    "john": dict(name="John", age=38, home="Chicago, USA", voice="мужской", sex="man",
                 piper="en_US-ryan-high", kokoro="am_michael"),
    "jane": dict(name="Jane", age=34, home="Seattle, USA", voice="женский", sex="woman",
                 piper="en_US-lessac-high", kokoro="af_heart"),
}
PERSONA_DEFAULT = "john"
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tutor_settings.json")


def load_persona_key():
    """Какого учителя выбрали в прошлый раз."""
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            key = json.load(f).get("persona")
        if key in PERSONAS:
            return key
    except Exception:
        pass
    return PERSONA_DEFAULT


def save_persona_key(key):
    try:
        with open(SETTINGS, "w", encoding="utf-8") as f:
            json.dump({"persona": key}, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


# --- словарь новых слов (память разговоров убрана) ---
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tutor_memory")
VOCAB_PATH = os.path.join(MEMORY_DIR, "vocabulary.json")
VOCAB_IN_PROMPT = 60       # сколько последних слов показывать учителю в промпте

TTS_ENGINE = "piper"             # "kokoro" или "piper"
KOKORO_LANG = "a"                # "a" - американский, "b" - британский английский
KOKORO_DEVICE = "cpu"
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper_voices")
RUSSIAN_VOICE = "ru_RU-irina-medium"   # для проговаривания русских слов в тренажёре

# --- VAD и перебивание ---
BLOCK = 512                # Silero на 16 кГц принимает ровно 512 отсчётов (32 мс)
VAD_THRESHOLD = 0.5        # выше - строже (меньше ложных срабатываний на шум)
SILENCE_MS = 700           # пауза, после которой фраза считается законченной
PREROLL_BLOCKS = 10        # ~0.3 с звука ДО начала речи, чтобы не обрезать первое слово
MIN_UTTER_S = 0.5          # фразы короче игнорируются (щелчки, кашель)
BARGE_IN_DEFAULT = False   # перебивание выключено по умолчанию (включайте в наушниках)
BARGE_MIN_S = 0.35         # сколько нужно говорить, чтобы прервать ответ
BARGE_MIN_RMS = 0.012      # ниже этой громкости микрофон не считается голосом (щелчки, шум)
ECHO_TAIL_S = 0.5          # после озвучки микрофон ещё чуть глушится (без перебивания)

# --- диагностика: пишется в tutor_debug.log рядом со скриптом (помогает разбирать обрывы звука) ---
DEBUG_LOG = True
DEBUG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tutor_debug.log")

# --- «учебный план»: программа сама ротирует времена и сценарии, чтобы занятия не повторялись ---
FOCUS_TURNS = 6            # сколько ваших реплик длится один фокус (время глагола + сценарий)
TENSES = [
    ("Present Simple", "habits, routines, facts: 'I usually...', 'How often do you...?'"),
    ("Present Continuous", "what is happening right now: 'I'm doing...', 'What are you working on?'"),
    ("Past Simple", "finished events: 'I went...', 'What did you do yesterday?'"),
    ("Past Continuous", "'was doing when...': interrupted actions and background scenes"),
    ("Present Perfect", "experience and results: 'Have you ever...?', 'I've just...'"),
    ("Present Perfect Continuous", "'How long have you been...?', 'I've been learning...'"),
    ("Future: will / going to", "plans and predictions: 'I'm going to...', 'What will you do...?'"),
    ("Used to", "past habits that are gone: 'I used to...', 'Did you use to...?'"),
    ("Past Perfect", "'I had done it before...': the earlier of two past events"),
    ("First Conditional", "real future possibilities: 'If you..., you will...'"),
    ("Second Conditional", "imaginary situations: 'If I were you...', 'What would you do if...?'"),
    ("Modals: can / could / should / must", "ability, advice, rules, obligation"),
    ("Comparatives and superlatives", "comparing: 'better than', 'the best', 'as ... as'"),
    ("Gerund and infinitive", "'I like doing', 'I want to do', 'I'm good at ...ing'"),
]
SCENARIOS = [
    "small talk in a Chicago coffee shop",
    "checking in at the airport and going through security",
    "at the pharmacy asking for something for a cold",
    "visiting a doctor about a bad cold",
    "checking into a hotel and asking about the city",
    "ordering food in a restaurant (and politely sending something back)",
    "a phone call to book a table",
    "making weekend plans with a friend",
    "remembering childhood and school days",
    "telling the story of a memorable trip",
    "shopping for clothes and returning an item",
    "talking to a landlord about renting an apartment",
    "gym, sport and healthy habits",
    "cooking dinner and sharing a recipe",
    "meeting a new neighbour for the first time",
    "talking about films and series you have watched",
    "a taxi ride across town",
    "a course or university: studies and exams",
    "weather, seasons and small talk about the forecast",
    "a phone call with a friend who lives abroad",
]
# Живые темы для болтовни: не сценарий, а то, о чём реально интересно трепаться.
TOPICS = [
    "your morning routine and how you get ready for the day",
    "commuting: buses, trains, traffic and being late",
    "groceries, supermarkets and what's in your fridge",
    "cooking at home: quick meals and kitchen disasters",
    "cleaning, laundry and keeping the flat tidy",
    "rent, bills and the joys of paying for utilities",
    "the doctor, the pharmacy and minor health problems",
    "weather, seasons and what you wear when it rains",
    "neighbours, noise and life in an apartment building",
    "phone calls, messages and staying in touch",
    "the bank, cards, cash and online payments",
    "shopping for clothes, sizes and returns",
    "public transport and getting around the city",
    "weekends: errands, rest and seeing friends",
    "food delivery, cafes and eating out",
    "sleep, alarms and being tired in the morning",
    "pets, plants and small home responsibilities",
    "plans for the evening and what you do to relax",
]
EVERYDAY_WORDS = [
    "commute", "rent", "bill", "receipt", "queue", "laundry", "groceries",
    "landlord", "neighbour", "appointment", "pharmacy", "prescription",
    "delivery", "leftovers", "kettle", "fridge", "sink", "tap", "bin",
    "charger", "wallet", "keys", "umbrella", "traffic jam", "bus stop",
    "train station", "checkout", "discount", "refund", "cash", "change",
    "queue up", "run errands", "do the washing-up", "take out the rubbish",
    "pay the bills", "book an appointment", "miss the bus", "be late for work",
]

# Банк слов для тренажёра: если словарь пуст или в нём мало слов, добираем отсюда.
WORD_BANK = [
    ("commute", "ездить на работу"),
    ("rent", "арендная плата"),
    ("bill", "счёт"),
    ("receipt", "чек"),
    ("queue", "очередь"),
    ("laundry", "стирка"),
    ("groceries", "продукты"),
    ("landlord", "арендодатель"),
    ("neighbour", "сосед"),
    ("appointment", "запись на приём"),
    ("pharmacy", "аптека"),
    ("prescription", "рецепт врача"),
    ("delivery", "доставка"),
    ("leftovers", "остатки еды"),
    ("kettle", "чайник"),
    ("fridge", "холодильник"),
    ("sink", "раковина"),
    ("tap", "кран"),
    ("bin", "мусорное ведро"),
    ("charger", "зарядное устройство"),
    ("wallet", "кошелёк"),
    ("keys", "ключи"),
    ("umbrella", "зонт"),
    ("traffic jam", "пробка"),
    ("bus stop", "автобусная остановка"),
    ("train station", "железнодорожный вокзал"),
    ("checkout", "касса"),
    ("discount", "скидка"),
    ("refund", "возврат денег"),
    ("cash", "наличные"),
    ("change", "сдача"),
    ("queue up", "стоять в очереди"),
    ("run errands", "делать поручения"),
    ("do the washing-up", "мыть посуду"),
    ("take out the rubbish", "выносить мусор"),
    ("pay the bills", "оплачивать счета"),
    ("book an appointment", "записаться на приём"),
    ("miss the bus", "опоздать на автобус"),
    ("be late for work", "опоздать на работу"),
    ("dye", "красить"),
    ("borrow", "брать взаймы"),
    ("lend", "давать взаймы"),
    ("owe", "быть должным"),
    ("afford", "позволить себе"),
    ("complain", "жаловаться"),
    ("apologise", "извиняться"),
    ("recommend", "рекомендовать"),
    ("remind", "напоминать"),
    ("realise", "осознавать"),
    ("manage", "успевать, справляться"),
]


def parse_word_pairs(text):
    """Достаёт из ответа модели пары «английское — русское»."""
    m = re.search(r"(?im)^\s*WORDS?\s*:\s*$", text)
    block = text[m.end():] if m else text
    pairs = []
    for line in block.splitlines():
        line = line.strip().lstrip("-•*0123456789. ").strip()
        if not line:
            continue
        if re.match(r"(?i)^(tutor|learner|teacher|student|you|me)\s*:", line):
            continue
        parts = re.split(r"\s+[—–-]\s+|\s*:\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        word, translation = parts[0].strip(), parts[1].strip()
        if not word or not translation:
            continue
        if re.match(r"(?i)^(tutor|learner|teacher|student|you|me)\s*:", word):
            continue
        if not re.search(r"[A-Za-z]", word):
            continue
        if not re.search(r"[А-Яа-яЁё]", translation):
            continue
        pairs.append((word, translation))
    return pairs


# --- режим «Изучение времён»: 12 времён, по одной русской фразе на каждое ---
TENSE_DRILLS = [
    ("Present Continuous", "Я иду на работу прямо сейчас.",
     "I'm going to work right now.",
     "am / is / are + verb-ing — She is cooking dinner right now."),
    ("Future Simple (will)", "Я пойду на работу завтра в 7 утра.",
     "I'll go to work tomorrow at 7 am.",
     "will + base verb — I'll call you later."),
    ("Past Continuous", "Когда я шёл на работу, ко мне подошёл полицейский.",
     "When I was walking to work, a policeman came up to me.",
     "was / were + verb-ing — They were watching TV at 8."),
    ("Present Simple", "Я обычно хожу на работу пешком.",
     "I usually walk to work.",
     "base verb / verb + -s — He works in a bank."),
    ("Past Simple", "Вчера я ходил на работу пешком.",
     "Yesterday I walked to work.",
     "verb + -ed / irregular form — She visited her parents yesterday."),
    ("Present Perfect", "Я уже закончил работу.",
     "I've already finished work.",
     "have / has + past participle — We have just eaten."),
    ("Present Perfect Continuous", "Я работаю здесь уже три года.",
     "I've been working here for three years.",
     "have / has been + verb-ing — He has been reading for an hour."),
    ("Future (going to)", "Я собираюсь пойти на работу завтра.",
     "I'm going to go to work tomorrow.",
     "am / is / are going to + base verb — We're going to move next month."),
    ("Past Perfect", "Когда я пришёл, она уже ушла.",
     "When I arrived, she had already left.",
     "had + past participle — The train had left before we arrived."),
    ("First Conditional", "Если пойдёт дождь, я останусь дома.",
     "If it rains, I'll stay at home.",
     "if + present simple, will + base verb — If you hurry, you'll catch the bus."),
    ("Second Conditional", "Если бы я был тобой, я бы не пошёл.",
     "If I were you, I wouldn't go.",
     "if + past simple, would + base verb — If I had more time, I would cook."),
    ("Used to", "Раньше я ходил на работу пешком.",
     "I used to walk to work.",
     "used to + base verb — I used to live in a small town."),
]
PRAISE = [
    ("Yes.", "(Да.)"),
    ("Good.", "(Хорошо.)"),
    ("Right.", "(Верно.)"),
    ("Nice.", "(Отлично.)"),
    ("Yep.", "(Ага.)"),
]

SYSTEM_PROMPT = """You are {name}, {age}, an English teacher who lives in {home}. You are a {sex}. You are warm, funny, a little sarcastic in a friendly way, and right now you are having a relaxed one-on-one conversation lesson with a Russian-speaking learner who needs to SPEAK English out loud. You are not an assistant and not a textbook: you are a real person sitting across the table with a coffee, and this is a chat between friends who happen to be practising English.

HOW YOU TALK - sound like a real person, not a chatbot
- 1-3 short sentences per turn, in natural spoken American English at A2-B1. Use contractions, filler words ("well", "honestly", "you know"), and real reactions ("Oh really?", "No way!", "Ha, that's wild.", "Ugh, I hate that.").
- Have opinions. Disagree sometimes, tease gently, crack a joke, tell a tiny story from your own life in {home}. Get curious about the weird details, not just the facts.
- Always finish with exactly ONE question. Make it specific, a bit unexpected, and growing out of the learner's last answer - not out of a new topic ("What was the weirdest part?", "Would you do it again?", "What did your mum say?"). Never ask two questions in the same reply, not even in two separate lines, and never repeat a question you already asked.
- Follow the learner's energy: if they get excited, dig into that; if they go quiet, throw them an easier, funnier question.
- Bring up genuinely interesting stuff - strange news, food, travel disasters, films, dreams, money, pets, childhood, "what would you do if...". The LESSON FOCUS below is only the grammar and the scene; the conversation itself should feel like two people having fun, not an exam.
- Stay on one topic for several turns, then move on naturally. Never announce the topic or the grammar.

KEEP THE CONVERSATION GOING DEEP - do not jump topics
- When the learner tells you something, react to it first: repeat or paraphrase what they said, say what you think, add a short detail or a tiny story from your own life, maybe joke about it.
- Then ask ONE follow-up question about the SAME thing. Good follow-ups: "Why?", "How did that feel?", "What happened next?", "What did you do?", "Would you do it again?", "What was the weirdest part?"
- Stay on the same topic for 3-4 exchanges. Only move to a new topic when the current one is genuinely exhausted, and bridge naturally ("Speaking of that...").
- Never change the topic just because you have another question ready. The learner should feel the conversation going deeper, not jumping around.
- If the learner gives a short answer, do not change the topic: ask a smaller, easier follow-up about the same thing.

WHAT YOU TRAIN (the main lesson work)
- The LESSON FOCUS below says which grammar to practise and which scene to play. Never announce it - just steer the conversation so the learner is forced to build sentences in that tense.
- Ask open questions ("Where did you go?", "What have you been doing lately?", "What would you do in my place?") so that a one-word answer is impossible. If you get a short answer anyway, ask again in a friendly way.
- Act out the scene: you are the barista, the waiter, the doctor, the receptionist... and stay in character while it is fun.
- Talking about grammar is useless - make the learner USE it.

VOCABULARY - your word notebook
- The VOCABULARY NOTEBOOK below lists words the learner has already collected. Use them naturally from time to time, but never add them again.
- When a genuinely new and useful word or phrase comes up (or you teach one), add at the very end of your reply exactly one line:
Word: <english word or phrase> — <Russian translation>
- Only one Word line per reply, only if it is really new, and never for basic words like "go", "work" or "good". The app saves it to the learner's dictionary.

MISTAKES - the most important part of your job
- Before you answer, silently check the learner's last sentence: is the tense right? the verb form? the word order? the articles? the preposition? the agreement?
- If there is a real mistake, your ENTIRE reply is ONLY this, in this exact order, and absolutely nothing else:
Fix: <the learner's sentence with the mistake corrected - one short sentence>
(<the Russian translation of that corrected sentence>)
Почему: <why, in simple Russian, one short sentence, no grammar jargon>
Now say it once more, please.
(Теперь скажи это ещё раз, пожалуйста.)
- No reaction, no comment on what they said, no answer to their question, no new question, no praise. The conversation is on hold until the learner repeats the correct sentence.
- Typical mistakes of Russian speakers - look for these first: wrong tense (go instead of went), 'I am agree', 'she don't', 'I very like it', missing 'is/are', wrong preposition (in the weekend, depend of).
- Two examples. The learner says "Yesterday I go to the cinema." Your whole reply is:
Fix: Yesterday I went to the cinema.
(Вчера я ходил в кино.)
Почему: прошедшее время - нужно went, а не go.
Now say it once more, please.
(Теперь скажи это ещё раз, пожалуйста.)
Then the learner says "I am agree with you." Your whole reply is:
Fix: I agree with you.
(Я согласен с тобой.)
Почему: после agree не нужен am - просто I agree.
Now say it once more, please.
(Теперь скажи это ещё раз, пожалуйста.)
- Never invent mistakes and never fix a correct sentence: if everything was fine, write no Fix and no Почему lines at all.
- If the learner spoke Russian (or mixed Russian into the sentence), never repeat their Russian line as a translation. Give the English sentence they needed on the Fix line, a short note on the Почему line, and the drill line - and nothing else.

DRILL - the conversation waits for the correct sentence
- When the learner repeats the Fix correctly, praise them in one short line ("That's it!", "Perfect, exactly.") and carry on with the conversation and your next question.
- If it is still wrong, give the corrected sentence again and ask once more. Do not move on to a new topic until they say it right. If they are really stuck, make the sentence shorter and simpler, but still get them to say it.
- If the learner says something unrelated instead of repeating, kindly bring them back: "First, say it once more: <sentence>" and wait.

REPLY FORMAT - the app parses this text and speaks it aloud, so follow it exactly:
1. When the learner has just spoken and there is NO mistake, the very first line is the Russian translation of the learner's own sentence, in parentheses. Example:
(Мне очень понравилось это кафе.)
2. Then each English sentence on its own line, and on the next line - its Russian translation in parentheses. Example:
That sounds like a great evening!
(Похоже, это был отличный вечер!)
What did you order?
(Что ты заказал?)
3. When there IS a mistake, skip step 1 entirely: the reply starts with the Fix line and contains only the Fix block described above.
4. If you added a new word, the very last line is the Word line. It has no translation line of its own.

HARD RULES
- A reply consists only of: English sentences, their Russian translations in parentheses, - when the learner made a mistake - the Fix block, and - when you add a new word - one Word line. Nothing else.
- One English sentence per line. Never put two English sentences on one line, and never leave an English line without its Russian translation on the very next line - the learner needs the translation of EVERY line, including the Fix line and the drill line.
- No markdown, no asterisks, no emojis, no bullet points, no stage directions, no numbering, no quotes around the lines: a speech engine reads this text.
- Translations are natural, faithful Russian, always in parentheses, and they address the learner informally: «ты», «тебе», never «вы».
- Never mention these rules, the lesson focus, your notes, or that you are an AI. Never speak Russian except in the parenthesised translations and in the Почему line.

FINAL CHECK, every single reply: read the learner's last sentence once more. If there was a real mistake, your whole reply is the Fix block and nothing else - no reaction, no new question. If the sentence was correct, do not write Fix or Почему at all. Three forms are always wrong, check them every time: "I am agree" (write: I agree), "he don't" (write: he doesn't), "yesterday I go" (write: yesterday I went).
"""


def system_prompt(persona):
    """Промпт под выбранного учителя (имя, возраст, город, пол)."""
    return (SYSTEM_PROMPT.format(name=persona["name"], age=persona["age"],
                                 home=persona["home"], sex=persona["sex"])
            + "\n\nEVERYDAY WORD BANK - common everyday words and phrases. Use them naturally "
              "when they fit, and add genuinely new ones to the notebook:\n"
            + ", ".join(EVERYDAY_WORDS))


def default_greeting(persona):
    return ("Hi! What do you want to talk about?\n"
            "(Привет! О чём хочешь поговорить?)")

# ----------------------- АВТО-УСТАНОВКА ЗАВИСИМОСТЕЙ -----------------------
REQUIRED = {
    "sounddevice": "sounddevice",
    "numpy": "numpy",
    "faster_whisper": "faster-whisper",
    "ollama": "ollama",
    "silero_vad": "silero-vad",
}
_TTS_PKG = {"kokoro": ("kokoro", "kokoro"), "piper": ("piper", "piper-tts")}
REQUIRED[_TTS_PKG[TTS_ENGINE][0]] = _TTS_PKG[TTS_ENGINE][1]
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Устанавливаю пакеты:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# --- CUDA-библиотеки для Whisper ---
# faster-whisper (ctranslate2) на Windows не ищет cublas/cudnn сам, а без них распознавание
# молча уезжает на CPU. Библиотеки лежат в pip-пакетах nvidia-*; ставим их и добавляем
# их bin-папки в список поиска DLL (это надо сделать ДО импорта ctranslate2).
WHISPER_USE_CUDA = True      # False - принудительно распознавать на CPU
CUDA_PKGS = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]


def cuda_dll_dirs():
    """bin-папки пакетов nvidia-* внутри site-packages (там cublas64_12.dll и cudnn)."""
    dirs = []
    for p in sys.path:
        nv = os.path.join(p, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in sorted(os.listdir(nv)):
            binp = os.path.join(nv, sub, "bin")
            if os.path.isdir(binp):
                dirs.append(binp)
    return dirs


def enable_cuda_dlls(install=True):
    """Готовит DLL для GPU-распознавания. True - cublas на месте."""
    have = any("cublas64_12.dll" in os.listdir(d) for d in cuda_dll_dirs())
    if not have and install and shutil.which("nvidia-smi"):
        print("Ставлю CUDA-библиотеки для Whisper:", ", ".join(CUDA_PKGS), "(~1.2 ГБ, один раз)")
        subprocess.call([sys.executable, "-m", "pip", "install", *CUDA_PKGS])
    dirs = cuda_dll_dirs()
    for d in dirs:
        try:
            os.add_dll_directory(d)          # Windows: добавляет папку в поиск DLL
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return any("cublas64_12.dll" in os.listdir(d) for d in dirs)


CUDA_DLLS = WHISPER_USE_CUDA and enable_cuda_dlls()

import numpy as np          # noqa: E402
import ollama               # noqa: E402
import sounddevice as sd    # noqa: E402
import torch                # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from silero_vad import VADIterator, load_silero_vad  # noqa: E402

CYR = re.compile(r"[а-яёА-ЯЁ]")
SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")
BULLET_RE = re.compile(r"^[-–—*•·]\s+")
FIX_RE = re.compile(r"^(?:fix(?:ed)?|corrected|правильно|исправление|исправлено|верно)\b\s*[:\-–—]?\s*(.*)$",
                    re.I)
NOTE_RE = re.compile(r"^(?:почему|объяснение|потому что)\b\s*[:\-–—]?\s*(.*)$", re.I)
WORD_RE = re.compile(r"^(?:word|слово|новое слово)\b\s*[:\-–—]?\s*(.*)$", re.I)
RU_PREFIX_RE = re.compile(r"^(?:ru|rus|перевод)\s*[:\-–—]\s*(.*)$", re.I)
PAREN_SPLIT_RE = re.compile(r"^([^()]*)\(([^()]*)\)\s*(.*)$")
PAREN_RE = re.compile(r"\([^)]*\)")
MD_RE = re.compile(r"[*_`~]+")          # markdown-звёздочки и подчёркивания: их не видно и не слышно
DONE = object()

STATES = {
    "loading": ("Загрузка", "#6b7280"),
    "paused": ("Пауза", "#6b7280"),
    "listening": ("Слушаю", "#15803d"),
    "hearing": ("Слышу вас", "#1d4ed8"),
    "thinking": ("Думаю", "#c2410c"),
    "speaking": ("Говорю", "#7c3aed"),
}


def g(obj, key):
    """Достаёт поле и из dict, и из объектов новых версий ollama."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def drain(buf, final=False):
    """Режет поток текста на законченные предложения."""
    parts = SENT_END.split(buf)
    rest = "" if final else parts.pop()
    return [p.strip() for p in parts if p.strip()], rest


def mostly_russian(s):
    """Строка в основном на русском (перевод), а не на английском."""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if CYR.match(c)) / len(letters) > 0.5


def plain(s):
    """Убирает markdown-мусор: звёздочки и подчёркивания модель иногда ставит для «выделения»,
    а движок озвучки читает их вслух («астериск»)."""
    return MD_RE.sub("", s).replace("  ", " ").strip()


def dbg(*parts):
    """Строка в tutor_debug.log: по ней видно, что именно произошло со звуком."""
    if not DEBUG_LOG:
        return
    try:
        with open(DEBUG_PATH, "a", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%H:%M:%S.%f")[:-3] + "  " +
                    "  ".join(str(p) for p in parts) + "\n")
    except OSError:
        pass


def parenthesise(s):
    """Приводит перевод к виду (текст), чтобы он всегда был в скобках."""
    s = s.strip()
    while len(s) > 1 and s[0] in "\"'«" and s[-1] in "\"'»":
        s = s[1:-1].strip()
    if s.startswith("(") and s.endswith(")"):
        return s
    return "(" + s.strip("()").strip() + ")"


def focus_for(turn):
    """-> (время глагола, подсказка, сценарий, живая тема) для текущей реплики ученика."""
    i = turn // FOCUS_TURNS
    tense, hint = TENSES[i % len(TENSES)]
    return tense, hint, SCENARIOS[i % len(SCENARIOS)], TOPICS[i % len(TOPICS)]


def focus_block(turn):
    tense, hint, scenario, topic = focus_for(turn)
    return ("\n\nLESSON FOCUS for this part of the conversation (never announce it, just steer the talk):\n"
            f"- grammar to practise: {tense} ({hint})\n"
            f"- scene to play out now: {scenario}\n"
            f"- a topic to explore only after the current scene is exhausted: {topic}\n"
            "Stay in this scene for several turns. React to the learner's answer, add your own thought "
            "or a tiny story, and ask a follow-up about the same thing. Only move on when the current "
            "topic is genuinely exhausted.")


def focus_label(turn):
    tense, _hint, scenario, _topic = focus_for(turn)
    return f"{tense}  ·  {scenario}"


def drill_block(sentence):
    """Проверка повтора: ученик должен был сказать исправленную фразу, разговор ждёт."""
    if not sentence:
        return ""
    return ("\n\nDRILL CHECK - handle this before anything else:\n"
            f"In your previous turn you corrected the learner and asked them to repeat: \"{sentence}\"\n"
            "The learner has just spoken. If they repeated it correctly (small pronunciation slips are fine), "
            "react with ONE short warm line and then continue the conversation with ONE new question - do not "
            "mention the correction again. If it is still wrong, reply with ONLY the Fix block (corrected "
            "sentence, its translation, Почему, and 'Now say it once more, please.' with its translation) and "
            "nothing else. If they said something unrelated instead of repeating, kindly ask them to say the "
            "sentence first, and nothing else.")


TENSE_EVAL_PROMPT = """You are {name}, a friendly English teacher checking one answer in a tense drill.
The learner had to translate this Russian sentence into English using {tense}:
Russian: «{russian}»
A correct answer looks like: «{english}»
{repeat_note}
Judge the learner's answer. Small variations, contractions and natural synonyms are fine - do not be pedantic.
- If the answer is correct, reply with exactly one line: OK
- If it is wrong, reply with ONLY this block, nothing else:
Fix: <the correct English sentence>
(<its Russian translation>)
Почему: <one short sentence in simple Russian, no grammar jargon>
Say it again.
Never invent mistakes. Never add a reaction, a comment or a new question.
"""


def tense_eval_prompt(persona, tense, russian, english, repeat):
    repeat_note = ("The learner is repeating your correction. If they now say it correctly, reply OK."
                   if repeat else "")
    return TENSE_EVAL_PROMPT.format(name=persona["name"], tense=tense, russian=russian,
                                    english=english, repeat_note=repeat_note)


TENSE_DRILL_PROMPT = """You are an English teacher preparing a fresh tense drill for a Russian-speaking learner.
Write TWO new, everyday Russian sentences for each tense below, with natural English translations and very short hints.
Use exactly this format, one line per sentence, fields separated by " || ":
Tense || Russian sentence || English sentence || hint

Tenses:
{tenses}

Rules:
- Write exactly two lines per tense.
- The Russian sentence must clearly require the named tense.
- The English sentence must be natural and correct.
- The hint is a short formula, e.g. "am / is / are + verb-ing".
- No numbering, no extra text, no blank lines.
"""


def generate_tense_drills():
    """Свежая порция фраз для тренажёра времён; при сбое - статический список."""
    expected = [d[0] for d in TENSE_DRILLS]
    prompt = TENSE_DRILL_PROMPT.format(tenses="\n".join(f"- {t}" for t in expected))
    try:
        text = ask(prompt, temperature=0.8)
    except Exception:
        return list(TENSE_DRILLS)
    by_tense = {}
    for line in text.splitlines():
        if "||" in line:
            parts = [p.strip() for p in line.split("||")]
        elif "|" in line:
            parts = [p.strip() for p in line.split("|")]
        else:
            continue
        if len(parts) < 4:
            continue
        name = parts[0].strip().strip("0123456789.-) ").lower()
        by_tense.setdefault(name, []).append(parts[1:4])
    drills = []
    for i, tense in enumerate(expected):
        key = tense.lower()
        entries = by_tense.get(key, [])
        if not entries:
            for name, value in by_tense.items():
                if key in name or name in key:
                    entries = value
                    break
        added = 0
        for parts in entries:
            if added >= 2:
                break
            russian, english, hint = parts[0], parts[1], parts[2]
            if not re.search(r"[а-яё]", russian, re.I) and re.search(r"[a-z]", russian, re.I):
                russian, english = english, russian
            if re.search(r"[а-яё]", russian, re.I) and re.search(r"[a-z]", english, re.I):
                drills.append((tense, russian, english, hint or TENSE_DRILLS[i][3]))
                added += 1
        if added == 0:
            drills.append(TENSE_DRILLS[i])
    return drills


WORD_EVAL_PROMPT = """You are {name}, a friendly English teacher checking one vocabulary answer.
The learner had to translate this Russian word or phrase into English:
Russian: «{russian}»
A correct answer looks like: «{english}»
{repeat_note}
Judge the learner's answer. Small variations, synonyms and natural alternatives are fine - do not be pedantic.
- If the learner's answer is a homophone or sounds almost the same as the target (for example "die" for "dye", "sea" for "see", "flour" for "flower"), accept it as correct: the microphone may have misheard. Reply OK.
- If the answer is correct, reply with exactly one line: OK
- If it is wrong, reply with ONLY this block, nothing else:
Fix: <the correct English word or phrase>
(<its Russian translation>)
Почему: <one short sentence in simple Russian, no grammar jargon>
Say it again.
Never invent mistakes. Never add a reaction, a comment or a new question.
"""


def word_eval_prompt(persona, russian, english, repeat):
    repeat_note = ("The learner is repeating your correction. If they now say it correctly, reply OK."
                   if repeat else "")
    return WORD_EVAL_PROMPT.format(name=persona["name"], russian=russian,
                                   english=english, repeat_note=repeat_note)


class ReplyReader:
    """
    Разбирает поток ответа Джона на события для окна и для озвучки:
      ("user_ru", текст) - перевод ВАШЕЙ реплики (первая строка ответа);
      ("en", текст)      - английская фраза, только её и озвучиваем;
      ("ru", текст)      - перевод этой фразы (чёрным, без озвучки);
      ("fix", текст)     - исправление вашей фразы;
      ("note", текст)    - объяснение по-русски («Почему: ...»);
      ("word", текст)    - новое слово для словаря («word — перевод»).
    Модель иногда путает строки, поэтому парсер терпимый: он понимает и скобки,
    и русскую строку без скобок, и перевод, приклеенный к фразе в той же строке.
    """

    def __init__(self, has_learner_line=True):
        self.buf = ""
        self.saw_en = False
        self.learner_pending = has_learner_line

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            out.extend(self._events(line))
        return out

    def flush(self):
        out = self._events(self.buf) if self.buf.strip() else []
        self.buf = ""
        return out

    def _ru(self, s):
        if self.learner_pending and not self.saw_en:
            self.learner_pending = False
            return [("user_ru", s)]
        return [("ru", s)]

    def _events(self, raw):
        """Разбор строки -> события, уже очищенные от markdown."""
        return [(kind, plain(text)) for kind, text in self._parse(raw)]

    def _parse(self, raw):
        s = BULLET_RE.sub("", raw.strip()).strip()
        s = s.strip("*_`").strip()          # на случай, если модель всё же добавит markdown
        if not s:
            return []
        m = FIX_RE.match(s)
        if m and m.group(1).strip():
            self.saw_en = True
            return [("fix", m.group(1).strip())]
        m = NOTE_RE.match(s)
        if m and m.group(1).strip():
            return [("note", m.group(1).strip())]
        m = WORD_RE.match(s)
        if m and m.group(1).strip():
            return [("word", m.group(1).strip())]
        m = RU_PREFIX_RE.match(s)
        if m and m.group(1).strip():
            return self._ru(m.group(1).strip())
        # перевод в скобках - сам по себе или приклеенный к английской фразе в этой же строке
        m = PAREN_SPLIT_RE.match(s)
        if m and mostly_russian(m.group(2)) and (
                not m.group(1).strip() or re.search(r"[A-Za-z]", m.group(1))):
            head, inside, tail = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            events = []
            if head:
                self.saw_en = True
                events.append(("en", head))
            events.extend(self._ru("(" + inside + ")"))
            if tail:
                events.extend(self._events(tail))
            return events
        if mostly_russian(s):
            return self._ru(s)
        self.saw_en = True
        return [("en", s)]


# ----------------------------- OLLAMA -----------------------------
def ollama_up():
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ensure_ollama(status):
    if not ollama_up():
        status("Запускаю ollama serve...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
        except FileNotFoundError:
            raise RuntimeError("Ollama не установлена: https://ollama.com/download")
        for _ in range(60):
            if ollama_up():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Ollama не отвечает")

    names = {g(m, "model") or g(m, "name") for m in g(ollama.list(), "models") or []}
    want = OLLAMA_MODEL if ":" in OLLAMA_MODEL else OLLAMA_MODEL + ":latest"
    if want not in names:
        status(f"Скачиваю {OLLAMA_MODEL}...")
        for p in ollama.pull(OLLAMA_MODEL, stream=True):
            total, done = g(p, "total"), g(p, "completed")
            if total and done:
                status(f"Скачиваю {OLLAMA_MODEL}: {done * 100 // total}%")


# ------------------------------ WHISPER ------------------------------
WHISPER_DEVICE = "?"   # куда реально встало распознавание: cuda или cpu


def load_whisper():
    global WHISPER_DEVICE
    if CUDA_DLLS:
        try:
            m = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8_float16")
            segs, _ = m.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"))
            list(segs)  # прогрев и проверка, что CUDA-библиотеки на месте
            WHISPER_DEVICE = "cuda"
            return m
        except Exception as e:
            print("Whisper на GPU не запустился, работаю на CPU:", e)
    WHISPER_DEVICE = "cpu"
    return WhisperModel("small", device="cpu", compute_type="int8")


# ------------------------------- СЛОВАРЬ -------------------------------
def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def ask(prompt, temperature=0.2):
    """Разовый запрос к модели (для проверки ответов); те же OPTIONS, чтобы Ollama не перезагружала модель."""
    r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}],
                    stream=False, keep_alive=KEEP_ALIVE, options=dict(OPTIONS, temperature=temperature))
    return (g(g(r, "message"), "content") or "").strip()


class Vocabulary:
    """Словарь новых слов: слово, перевод, дата, повторы и ошибки."""

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.lock = threading.Lock()
        self.words = load_json(VOCAB_PATH, [])

    @staticmethod
    def _ensure_stats(item):
        item.setdefault("repeats", 0)
        item.setdefault("errors", 0)
        item.setdefault("last_seen", "")
        item.setdefault("next_due", "")
        item.setdefault("interval", 0)

    def add(self, text):
        """Добавляет строку вида 'word — перевод'. Возвращает (word, translation) или None."""
        text = text.strip()
        if not text:
            return None
        parts = re.split(r"\s+[—–-]\s+|\s*:\s*", text, maxsplit=1)
        word = parts[0].strip()
        translation = parts[1].strip() if len(parts) > 1 else ""
        if not word:
            return None
        key = word.lower()
        with self.lock:
            for item in self.words:
                if item.get("word", "").lower() == key:
                    item["count"] = item.get("count", 1) + 1
                    if translation:
                        item["translation"] = translation
                    self._ensure_stats(item)
                    save_json(VOCAB_PATH, self.words)
                    return word, translation
            item = {"word": word, "translation": translation,
                    "added": date.today().isoformat(), "count": 1}
            self._ensure_stats(item)
            self.words.append(item)
            save_json(VOCAB_PATH, self.words)
        return word, translation

    def ensure_min_words(self, min_count=5):
        """Добирает слова из WORD_BANK, если в словаре меньше min_count слов с переводом."""
        with self.lock:
            existing = {w.get("word", "").lower(): w for w in self.words}
            have = {k for k, w in existing.items() if w.get("translation")}
            if len(have) >= min_count:
                return 0
            added = 0
            for word, translation in WORD_BANK:
                key = word.lower()
                if key in existing:
                    if not existing[key].get("translation"):
                        existing[key]["translation"] = translation
                        have.add(key)
                        added += 1
                    continue
                item = {"word": word, "translation": translation,
                        "added": date.today().isoformat(), "count": 1}
                self._ensure_stats(item)
                self.words.append(item)
                existing[key] = item
                have.add(key)
                added += 1
                if len(have) >= min_count:
                    break
            if added:
                save_json(VOCAB_PATH, self.words)
            return added

    def record(self, word, correct):
        """Отмечает повтор слова. correct=True - верно, False - ошибка."""
        now = datetime.now()
        with self.lock:
            for item in self.words:
                if item.get("word", "").lower() == word.lower():
                    self._ensure_stats(item)
                    item["repeats"] += 1
                    item["last_seen"] = now.isoformat(timespec="seconds")
                    if correct:
                        if item["errors"] > 0:
                            item["interval"] = 1
                            item["next_due"] = (now + timedelta(days=1)).isoformat(timespec="seconds")
                        else:
                            interval = item["interval"]
                            interval = 1 if interval < 1 else min(interval * 2, 30)
                            item["interval"] = interval
                            item["next_due"] = (now + timedelta(days=interval)).isoformat(timespec="seconds")
                    else:
                        item["errors"] += 1
                        item["interval"] = 0
                        item["next_due"] = (now + timedelta(minutes=2)).isoformat(timespec="seconds")
                    save_json(VOCAB_PATH, self.words)
                    return item
        return None

    def skip(self, word):
        """Пропуск слова: не ошибка, но вернём его позже (через 10 минут)."""
        now = datetime.now()
        with self.lock:
            for item in self.words:
                if item.get("word", "").lower() == word.lower():
                    self._ensure_stats(item)
                    item["repeats"] += 1
                    item["last_seen"] = now.isoformat(timespec="seconds")
                    item["next_due"] = (now + timedelta(minutes=10)).isoformat(timespec="seconds")
                    save_json(VOCAB_PATH, self.words)
                    return item
        return None

    def next_due(self, exclude=None, seen=None):
        """Следующее слово по правилу забывания: сначала новые, потом просроченные."""
        with self.lock:
            pool = [w for w in self.words if w.get("translation")]
            if not pool:
                return None
            if exclude:
                others = [w for w in pool if w.get("word") != exclude]
                if not others:
                    return None          # в словаре только это слово - не повторяем его сразу
                pool = others
            now = datetime.now().isoformat(timespec="seconds")
            due = [w for w in pool if not w.get("next_due") or w["next_due"] <= now]
            pool = due or pool
            if seen:
                unseen = [w for w in pool if w.get("word") not in seen]
                if unseen:
                    pool = unseen

            def key(w):
                return (w.get("next_due") or "0000", w.get("repeats", 0))
            return min(pool, key=key)

    def prompt_block(self):
        if not self.words:
            return "\n\nVOCABULARY NOTEBOOK: empty. Start collecting useful new words."
        recent = self.words[-VOCAB_IN_PROMPT:]
        lines = "\n".join(
            f"- {w.get('word', '')} — {w.get('translation', '')}"
            + (f" (×{w.get('repeats', 0)})" if w.get("repeats", 0) else "")
            for w in recent)
        return ("\n\nVOCABULARY NOTEBOOK (words the learner has already collected; reuse them "
                "naturally, do not add them again):\n" + lines)

    def wipe(self):
        with self.lock:
            self.words = []
            save_json(VOCAB_PATH, self.words)


# -------------------------------- TTS --------------------------------
TTS_TAIL_S = 0.25          # тишина после каждой фразы: страховка от «откушенного» последнего слова
TTS_FADE_S = 0.015         # плавное начало и конец звука: без щелчка на краях фразы


class Speaker(threading.Thread):
    """
    Озвучка в отдельном потоке с поддержкой мгновенного прерывания.
    Каждый элемент очереди помечен номером «поколения» (gen): interrupt() увеличивает номер,
    и всё, что относится к старому ответу, выбрасывается.

    Каждая фраза играется целиком (sd.play + sd.wait), но перед этим к ней добавляются
    плавные края и хвост тишины: piper обрывает звук на ненулевом уровне, а поток устройства
    закрывается сразу после последнего сэмпла - без хвоста это слышно как «откусили слово».
    """

    def __init__(self, on_idle, status, piper_voice, kokoro_voice):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.gen = 0
        self.on_idle = on_idle
        self.status = status
        self.piper_voice = piper_voice
        self.kokoro_voice = kokoro_voice
        self.reload = False          # сменили учителя - перезагрузить голос
        self.start()

    def say(self, gen, text, lang="en"):
        self.q.put((gen, text, lang))

    def done(self, gen):
        self.q.put((gen, DONE, None))

    def set_voice(self, piper_voice, kokoro_voice):
        """Меняет голос: текущая фраза обрывается, движок перезагрузится на следующей."""
        self.piper_voice = piper_voice
        self.kokoro_voice = kokoro_voice
        self.reload = True
        self.interrupt()

    def interrupt(self):
        self.gen += 1
        dbg("speaker.interrupt -> gen", self.gen)
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        sd.stop()  # обрывает текущее воспроизведение (микрофонный поток не затрагивает)

    @staticmethod
    def shape(audio, rate):
        """Плавное начало и конец фразы: убирают щелчок, который слышен как обрыв слова."""
        a = np.ascontiguousarray(audio, dtype="float32")
        n = min(len(a) // 2, max(1, int(rate * TTS_FADE_S)))
        if n > 1:
            ramp = np.linspace(0.0, 1.0, n, dtype="float32")
            a[:n] *= ramp
            a[-n:] *= ramp[::-1]
        return a

    def play(self, blocks, alive):
        """blocks: итератор (звук float32, частота). Играет фразу с хвостом тишины."""
        for audio, sr in blocks:
            if not alive():
                return
            block = self.shape(audio, sr)
            tail = np.zeros(max(1, int(sr * TTS_TAIL_S)), dtype="float32")
            sd.play(np.concatenate([block, tail]), sr)
            sd.wait()

    # Каждый движок возвращает say(text, alive): alive() == False -> прекратить воспроизведение
    def _kokoro(self):
        from kokoro import KPipeline
        pipe = KPipeline(lang_code=KOKORO_LANG, device=KOKORO_DEVICE)

        def blocks(text):
            for _, _, audio in pipe(text, voice=self.kokoro_voice):
                if hasattr(audio, "detach"):  # torch.Tensor
                    audio = audio.detach().cpu().numpy()
                yield np.asarray(audio, dtype="float32"), 24000

        def say(text, alive):
            self.play(blocks(text), alive)
        return say

    def _piper(self):
        from piper import PiperVoice
        model = os.path.join(VOICES_DIR, self.piper_voice + ".onnx")
        if not os.path.exists(model):
            os.makedirs(VOICES_DIR, exist_ok=True)
            self.status(f"Скачиваю голос {self.piper_voice}...")
            subprocess.check_call([sys.executable, "-m", "piper.download_voices", self.piper_voice],
                                  cwd=VOICES_DIR)
        voice = PiperVoice.load(model)

        def blocks(text):
            for chunk in voice.synthesize(text):
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                yield audio.astype(np.float32) / 32768.0, chunk.sample_rate

        def say(text, alive):
            self.play(blocks(text), alive)
        return say

    def _piper_ru(self):
        """Русский голос для проговаривания слов в тренажёре."""
        from piper import PiperVoice
        model = os.path.join(VOICES_DIR, RUSSIAN_VOICE + ".onnx")
        if not os.path.exists(model):
            os.makedirs(VOICES_DIR, exist_ok=True)
            self.status(f"Скачиваю русский голос {RUSSIAN_VOICE}...")
            subprocess.check_call([sys.executable, "-m", "piper.download_voices", RUSSIAN_VOICE],
                                  cwd=VOICES_DIR)
        voice = PiperVoice.load(model)

        def blocks(text):
            for chunk in voice.synthesize(text):
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                yield audio.astype(np.float32) / 32768.0, chunk.sample_rate

        def say(text, alive):
            self.play(blocks(text), alive)
        return say

    def engine(self):
        """Загружает движок озвучки под текущий голос; при неудаче - второй движок."""
        for engine in dict.fromkeys([TTS_ENGINE, "piper" if TTS_ENGINE == "kokoro" else "kokoro"]):
            try:
                self.status(f"Загружаю голос ({engine})...")
                return getattr(self, "_" + engine)()
            except Exception as e:
                self.status(f"{engine} не запустился: {e}")
        self.status("Озвучка недоступна - работаю только текстом")
        return lambda text, alive: None

    def run(self):
        say = self.engine()
        ru_say = None
        while True:
            gen, item, lang = self.q.get()
            if gen != self.gen:
                continue  # устаревшее (прерванное) поколение
            if self.reload:                 # выбрали другого учителя - другой голос
                self.reload = False
                say = self.engine()
            if item is DONE:
                self.on_idle(gen)
            else:
                dbg("say:", item[:70])
                try:
                    if lang == "ru":
                        if ru_say is None:
                            try:
                                ru_say = self._piper_ru()
                            except Exception as e:
                                self.status(f"Русский голос не запустился: {e}")
                                ru_say = False
                        if ru_say:
                            ru_say(item, lambda gen=gen: gen == self.gen)
                        else:
                            say(item, lambda gen=gen: gen == self.gen)
                    else:
                        say(item, lambda gen=gen: gen == self.gen)
                except Exception as e:   # одна сбойная фраза не должна убивать озвучку целиком
                    dbg("say ERROR:", type(e).__name__, e)
                    self.status(f"Озвучка: {e}")
                dbg("say end")


# -------------------------------- GUI --------------------------------
FONT = "Segoe UI"          # системный шрифт без засечек
FONT_BODY = 14             # размер английских реплик
FONT_SUB = 13              # размер переводов и пояснений
BG = "#eef2f6"             # фон окна
CARD = "#ffffff"           # карточка диалога
INK = "#101418"
MUTED = "#66707c"
ME = "#1d4ed8"             # цвет ученика
TUTOR = "#0f7a45"          # цвет Джона
DANGER = "#b3261e"


def enable_hi_dpi():
    """Чёткие шрифты на мониторах с масштабированием Windows. Вызывать до tk.Tk()."""
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def load_avatar(key, px):
    """Круглый портрет учителя (avatar_<key>.png), уменьшенный до нужного размера."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"avatar_{key}.png")
    if not os.path.exists(path):
        return None
    try:
        img = tk.PhotoImage(file=path)
    except (tk.TclError, OSError):
        return None
    k = max(1, round(img.width() / px))
    return img.subsample(k, k) if k > 1 else img


class Tooltip:
    """Простая всплывающая подсказка для Tk-виджета."""
    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def set(self, text):
        self.text = text

    def show(self, _event=None):
        if not self.text or self.tip:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#fffbeb",
                 foreground=INK, relief="solid", borderwidth=1,
                 font=(FONT, 9), padx=8, pady=4, wraplength=420).pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App:
    def __init__(self, root):
        self.root = root
        self.persona_key = load_persona_key()
        self.persona = PERSONAS[self.persona_key]
        root.title(f"{self.persona['name']} · English tutor")
        root.configure(bg=BG)
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0)
        w = min(int(780 * self.scale), int(root.winfo_screenwidth() * 0.94))
        h = min(int(940 * self.scale), int(root.winfo_screenheight() * 0.92))
        root.geometry(f"{w}x{h}")
        root.minsize(min(w, int(540 * self.scale)), min(h, int(640 * self.scale)))

        self.ui = queue.Queue()
        self.audio_q = queue.Queue()   # блоки звука с микрофона
        self.utt_q = queue.Queue()     # готовые фразы пользователя
        self.cancel = threading.Event()  # прервать текущий ответ
        self.messages = []
        self.state = "loading"
        self.enabled = True
        self.mute_until = 0.0
        self.barge_flag = BARGE_IN_DEFAULT
        self.whisper = self.vad_model = self.speaker = self.mic = None
        self.vocab = None
        self.vocab_block = ""
        self.turn = 0                   # номер вашей реплики: по нему ротируется фокус занятия
        self.mode = "chat"              # "chat", "tenses", "words" или "lookup"
        self.base_mode = "chat"         # режим, в который вернёмся после «Спросить слово»
        self.tense_i = 0                # какое из 12 времён сейчас спрашиваем
        self.tense_drills = list(TENSE_DRILLS)   # фразы текущего захода (могут быть свежими)
        self.tense_current = None       # фраза, которую спросили прямо сейчас
        self.tense_correct = 0          # сколько фраз подряд угадано (для редкой похвалы)
        self._tense_loading = False     # сейчас генерируются новые фразы
        self.barge_ready = False        # перебивание включаем только после первого приветствия
        self.hold = False               # кнопка «Говорить» удерживается прямо сейчас
        self.hold_release = False       # кнопку отпустили - фразу пора отдать в распознавание
        self._mark_n = 0                # счётчик меток для вставки недостающих переводов
        self._pending = []              # английские строки, которым ещё не пришёл перевод
        self._tutor_started = False     # реплика Джона уже начата (напечатано имя)
        self._user_ru_shown = True      # перевод вашей реплики уже показан
        self._last_fix = None           # исправление из текущего ответа
        self._question_seen = False     # в этой реплике уже был вопрос
        self._skip_next_ru = False      # пропустить перевод отброшенного второго вопроса
        self.drill = None               # фраза, которую ученик должен повторить, прежде чем продолжим
        self.word_drill = None          # слово, которое ученик повторяет в режиме слов
        self.word_current = None        # текущее слово в тренажёре слов
        self.last_word = None           # предыдущее слово, чтобы не спрашивать его дважды подряд
        self.word_attempts = 0          # сколько раз подряд не получилось текущее слово
        self.word_correct = 0           # сколько слов подряд угадано (для редкой похвалы)
        self.word_hint_level = 0        # 0 - без подсказки, 1 - первая буква, 2 - две буквы, 3 - всё слово

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=(FONT, 10), background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("TButton", font=(FONT, 10), padding=(16, 9), relief="flat",
                        borderwidth=0, background=CARD, foreground=INK)
        style.map("TButton", background=[("active", "#e6edf6"), ("pressed", "#d8e2ee")])
        style.configure("Accent.TButton", background="#1f6feb", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#2b7bf3"), ("pressed", "#1b61d1")])
        style.configure("Talk.TButton", font=(FONT, 12, "bold"), padding=(16, 11),
                        background="#0f7a45", foreground="#ffffff")
        style.map("Talk.TButton", background=[("active", "#128a4e"), ("pressed", "#0a5c34")])
        style.configure("TCheckbutton", background=BG, foreground=MUTED)
        style.configure("Danger.TButton", foreground=DANGER)
        style.configure("Mode.TButton", font=(FONT, 9), padding=(10, 6))
        style.configure("TenseOn.TButton", font=(FONT, 9), padding=(10, 6),
                        background="#7c3aed", foreground="#ffffff")
        style.map("TenseOn.TButton", background=[("active", "#6d28d9"), ("pressed", "#5b21b6")])
        style.configure("ChatOn.TButton", font=(FONT, 9), padding=(10, 6),
                        background="#1f6feb", foreground="#ffffff")
        style.map("ChatOn.TButton", background=[("active", "#2b7bf3"), ("pressed", "#1b61d1")])
        style.configure("WordsOn.TButton", font=(FONT, 9), padding=(10, 6),
                        background="#0f7a45", foreground="#ffffff")
        style.map("WordsOn.TButton", background=[("active", "#128a4e"), ("pressed", "#0a5c34")])
        style.configure("LookupOn.TButton", font=(FONT, 9), padding=(10, 6),
                        background="#c2410c", foreground="#ffffff")
        style.map("LookupOn.TButton", background=[("active", "#9a3412"), ("pressed", "#7c2d12")])
        style.configure("TalkSmall.TButton", font=(FONT, 10, "bold"), padding=(14, 7),
                        background="#0f7a45", foreground="#ffffff")
        style.map("TalkSmall.TButton", background=[("active", "#128a4e"), ("pressed", "#0a5c34")])
        style.configure("TCombobox", font=(FONT, 10), padding=(8, 4),
                        fieldbackground=CARD, background=CARD, foreground=INK,
                        arrowcolor=MUTED, bordercolor="#dbe2ea", lightcolor=CARD,
                        darkcolor=CARD)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)],
                  selectbackground=[("readonly", CARD)], selectforeground=[("readonly", INK)])
        style.configure("Vertical.TScrollbar", background="#b8c4d4", troughcolor=CARD,
                        borderwidth=0, arrowcolor=CARD, arrowsize=10, width=10)
        style.map("Vertical.TScrollbar", background=[("active", "#9fb0c6")])
        # скроллбар без уродливых стрелок - только ползунок
        style.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {
            "sticky": "ns", "children": [("Vertical.Scrollbar.thumb",
                                          {"expand": "1", "sticky": "nswe"})]})])

        # ---- шапка ----
        head = ttk.Frame(root, padding=(18, 10, 18, 4))
        head.pack(fill="x")
        self.avatar_label = tk.Label(head, bg=BG)
        self.avatar_label.pack(side="left", padx=(0, 12))
        titles = ttk.Frame(head)
        titles.pack(side="left")
        self.name_label = tk.Label(titles, text=self.persona["name"], font=(FONT, 22, "bold"),
                                   bg=BG, fg=INK, anchor="w")
        self.name_label.pack(anchor="w")
        self.sub_label = tk.Label(titles, text=f"English tutor  ·  {self.persona['home']}",
                                  font=(FONT, 10), bg=BG, fg=MUTED, anchor="w")
        self.sub_label.pack(anchor="w")
        self.pill = tk.Label(head, text="Загрузка", font=(FONT, 10, "bold"), bg=MUTED,
                             fg="#ffffff", padx=14, pady=7)
        self.pill.pack(side="right")
        self.update_avatar()

        # ---- карточка диалога ----
        body = ttk.Frame(root, padding=(18, 6, 18, 6))
        body.pack(fill="both", expand=True)
        card = tk.Frame(body, bg=CARD, highlightbackground="#dbe2ea",
                        highlightcolor="#dbe2ea", highlightthickness=1)
        card.pack(fill="both", expand=True)
        self.text = tk.Text(card, wrap="word", font=(FONT, FONT_BODY),
                            bg=CARD, fg=INK, relief="flat", borderwidth=0, highlightthickness=0,
                            padx=20, pady=16, insertwidth=0, cursor="xterm",
                            spacing2=2, selectbackground="#cfe0ff", selectforeground=INK)
        sb = ttk.Scrollbar(card, command=self.text.yview)
        self.text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", pady=8, padx=(0, 6))
        self.text.pack(side="left", fill="both", expand=True)

        # Окно только для чтения, но с рабочим выделением и копированием:
        # disabled-Text в Tk не позволяет выделять текст, поэтому блокируем ввод вручную.
        self.text.bind("<Key>", self._readonly_key)
        self.text.bind("<Control-c>", self._copy_key)
        self.text.bind("<Control-C>", self._copy_key)
        self.text.bind("<Control-a>", self._select_all_key)
        self.text.bind("<Control-A>", self._select_all_key)
        self.menu = tk.Menu(root, tearoff=0, font=(FONT, 10))
        self.menu.add_command(label="Копировать", command=self.copy_selection)
        self.menu.add_command(label="Копировать весь диалог", command=self.copy_all)
        self.menu.add_separator()
        self.menu.add_command(label="Выделить всё", command=self.select_all)
        self.text.bind("<Button-3>", self._popup_menu)
        for seq in ("<Control-v>", "<Control-V>", "<Shift-Insert>", "<Button-2>"):
            self.text.bind(seq, lambda e: "break")   # вставка в окно запрещена

        t = self.text
        t.tag_configure("name_jn", font=(FONT, 11, "bold"), foreground=TUTOR,
                        spacing1=22, spacing3=5)
        t.tag_configure("name_me", font=(FONT, 11, "bold"), foreground=ME,
                        spacing1=22, spacing3=5)
        t.tag_configure("en_jn", font=(FONT, FONT_BODY), foreground="#123d2c",
                        lmargin1=14, lmargin2=14, spacing1=4, spacing3=4)
        t.tag_configure("en_me", font=(FONT, FONT_BODY), foreground="#173a63",
                        lmargin1=14, lmargin2=14, spacing1=4, spacing3=4)
        t.tag_configure("ru", font=(FONT, FONT_SUB), foreground="#000000",
                        lmargin1=28, lmargin2=28, spacing3=10)
        t.tag_configure("fix", font=(FONT, FONT_SUB, "bold"), foreground=DANGER,
                        lmargin1=14, lmargin2=14, spacing1=12, spacing3=4)
        t.tag_configure("note", font=(FONT, FONT_SUB), foreground="#000000",
                        lmargin1=14, lmargin2=14, spacing3=10)
        t.tag_configure("task", font=(FONT, FONT_BODY, "bold"), foreground="#7c3aed",
                        lmargin1=14, lmargin2=14, spacing1=8, spacing3=8)
        t.tag_configure("word", font=(FONT, FONT_SUB, "bold"), foreground="#0f766e",
                        lmargin1=14, lmargin2=14, spacing1=8, spacing3=8)
        t.tag_configure("hint", font=(FONT, FONT_SUB), foreground="#6b7280",
                        lmargin1=14, lmargin2=14, spacing3=10)
        t.tag_configure("stop", font=(FONT, 11), foreground="#9aa4b2", spacing1=4)

        # кнопка «Говорить» теперь в ряду режимов

        # ---- подпись и фокус занятия ----
        info = ttk.Frame(root, padding=(20, 6, 20, 0))
        info.pack(fill="x")
        self.status_var = tk.StringVar(value="Запуск...")
        tk.Label(info, textvariable=self.status_var, font=(FONT, 9), bg=BG, fg=MUTED,
                 anchor="w").pack(side="left")
        self.focus_var = tk.StringVar(value="")
        self.focus_label = tk.Label(info, textvariable=self.focus_var, font=(FONT, 9),
                                    bg=BG, fg="#2563eb", anchor="e")
        self.focus_label.pack(side="right")
        self.focus_tip = Tooltip(self.focus_label, "")

        # ---- кнопки ----
        bar = ttk.Frame(root, padding=(18, 8, 18, 12))
        bar.pack(fill="x")
        row = ttk.Frame(bar)
        row.pack(fill="x")
        self.toggle_btn = ttk.Button(row, text="Пауза", style="Accent.TButton",
                                     command=self.toggle)
        self.toggle_btn.pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Прервать", command=self.manual_interrupt).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Новый диалог", command=self.reset).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Стереть словарь", style="Danger.TButton",
                   command=self.wipe_vocab).pack(side="left")
        self.barge_var = tk.BooleanVar(value=BARGE_IN_DEFAULT)
        tk.Checkbutton(row, text="Перебивание (в наушниках)", variable=self.barge_var,
                       command=lambda: setattr(self, "barge_flag", self.barge_var.get()),
                       font=(FONT, 9), bg=BG, fg=MUTED, activebackground=BG,
                       activeforeground=INK, selectcolor=CARD, highlightthickness=0,
                       borderwidth=0, cursor="hand2").pack(side="right")

        # ---- выбор учителя ----
        pick = ttk.Frame(bar)
        pick.pack(fill="x", pady=(8, 0))
        tk.Label(pick, text="Учитель:", font=(FONT, 9), bg=BG, fg=MUTED).pack(side="left")
        self._persona_items = {f"{p['name']} · {p['voice']} голос": key
                               for key, p in PERSONAS.items()}
        self.persona_box = ttk.Combobox(pick, state="readonly", width=24, font=(FONT, 10),
                                        values=list(self._persona_items))
        self.persona_box.set(f"{self.persona['name']} · {self.persona['voice']} голос")
        self.persona_box.pack(side="left", padx=(8, 0))
        self.persona_box.bind("<<ComboboxSelected>>", self._persona_chosen)
        tk.Label(pick, text="смена учителя начинает новый диалог", font=(FONT, 9),
                 bg=BG, fg=MUTED).pack(side="left", padx=(10, 0))

        # ---- режимы и кнопка «Говорить» в одном ряду ----
        modes = ttk.Frame(bar)
        modes.pack(fill="x", pady=(8, 0))
        tk.Label(modes, text="Режим:", font=(FONT, 9), bg=BG, fg=MUTED).pack(side="left")
        self.chat_btn = ttk.Button(modes, text="Обычный разговор", state="disabled",
                                   command=lambda: self.set_mode("chat"))
        self.chat_btn.pack(side="left", padx=(8, 0))
        self.tense_btn = ttk.Button(modes, text="Изучение времён", state="disabled",
                                    command=lambda: self.set_mode("tenses"))
        self.tense_btn.pack(side="left", padx=(8, 0))
        self.words_btn = ttk.Button(modes, text="Изучение слов", state="disabled",
                                    command=lambda: self.set_mode("words"))
        self.words_btn.pack(side="left", padx=(8, 0))
        self.lookup_btn = ttk.Button(modes, text="Спросить слово", state="disabled",
                                     command=lambda: self.set_mode("lookup"))
        self.lookup_btn.pack(side="left", padx=(8, 0))
        self.mode_btns = [self.chat_btn, self.tense_btn, self.words_btn, self.lookup_btn]
        self.talk_btn = ttk.Button(modes, text="Говорить — держите кнопку",
                                   style="TalkSmall.TButton")
        self.talk_btn.pack(side="right")
        self.talk_btn.bind("<ButtonPress-1>", self.start_hold)
        self.talk_btn.bind("<ButtonRelease-1>", self.stop_hold)

        # ---- инструменты тренажёра слов: подсказка и пропуск ----
        self.word_tools = ttk.Frame(bar)
        self.hint_btn = ttk.Button(self.word_tools, text="Подсказка",
                                   command=self.word_hint)
        self.hint_btn.pack(side="left")
        self.skip_btn = ttk.Button(self.word_tools, text="Следующее слово",
                                   command=self.word_skip)
        self.skip_btn.pack(side="left", padx=(8, 0))
        self.add_words_btn = ttk.Button(self.word_tools, text="Добавить слова",
                                        command=self.add_words)
        self.add_words_btn.pack(side="left", padx=(8, 0))
        tk.Label(self.word_tools, text="или скажи «Подсказка» / «Следующее слово»",
                 font=(FONT, 9), bg=BG, fg=MUTED).pack(side="left", padx=(10, 0))
        self.word_tools.pack_forget()
        self._update_mode_buttons()

        self.status = lambda t: self.ui.put(("status", t))
        threading.Thread(target=self.setup, daemon=True).start()
        root.after(50, self.poll)

    # ---- вывод в окно (из главного потока через очередь) ----
    def poll(self):
        try:
            while True:
                kind, *args = self.ui.get_nowait()
                if kind == "status":
                    self.status_var.set(args[0])
                elif kind == "text":
                    self.text.insert("end", args[0], tuple(args[1:]))
                    self.text.see("end")
                elif kind == "state":
                    label, color = STATES[args[0]]
                    self.pill.config(text=label, bg=color)
                elif kind == "focus":
                    self.focus_var.set(args[0])
                elif kind == "tense_ready":
                    for btn in self.mode_btns:
                        btn.config(state="normal")
                elif kind == "mark":
                    # gravity="left": метка остаётся ПЕРЕД текстом, который допишут ниже,
                    # то есть ровно в конце своей строки
                    self.text.mark_set(args[0], "end-1c")
                    self.text.mark_gravity(args[0], "left")
                elif kind == "insert_at":
                    mark, s, tag = args
                    try:
                        self.text.insert(mark, s, tag)
                    except tk.TclError:
                        pass
                    self.text.see("end")
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def say_text(self, s, *tags):
        self.ui.put(("text", s, *tags))

    def set_state(self, st):
        self.state = st
        self.ui.put(("state", st))

    # ---- окно только для чтения, но с рабочим выделением и копированием ----
    NAV_KEYS = {"left", "right", "up", "down", "home", "end", "prior", "next",
                "shift_l", "shift_r", "control_l", "control_r", "tab", "caps_lock", "escape"}

    def _readonly_key(self, event):
        """Выделение и навигация работают, ввод с клавиатуры - нет."""
        if (event.keysym or "").lower() in self.NAV_KEYS:
            return None
        return "break"

    def _copy_key(self, _event):
        self.copy_selection()
        return "break"

    def _select_all_key(self, _event):
        self.select_all()
        return "break"

    def select_all(self):
        self.text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def copy_selection(self):
        try:
            txt = self.text.get("sel.first", "sel.last")
        except tk.TclError:
            txt = ""
        if not txt.strip():
            self.status("Сначала выделите текст мышью (или Ctrl+A)")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status(f"Скопировано символов: {len(txt)}")

    def copy_all(self):
        txt = self.text.get("1.0", "end-1c").strip()
        if not txt:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status(f"Скопирован весь диалог: {len(txt)} символов")

    def _popup_menu(self, event):
        self.text.focus_set()
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
        return "break"

    # ---- выбор учителя: John или Jane ----
    def update_avatar(self):
        """Портрет учителя в шапке (если картинки нет - первая буква имени)."""
        img = load_avatar(self.persona_key, int(56 * self.scale))
        self._avatar_img = img
        if img:
            self.avatar_label.config(image=img, text="", width=0, height=0, bg=BG)
        else:
            self.avatar_label.config(image="", text=self.persona["name"][:1],
                                     font=(FONT, 20, "bold"), fg="#ffffff", bg=TUTOR,
                                     width=3, height=1)

    def _persona_chosen(self, _event=None):
        key = self._persona_items.get(self.persona_box.get())
        if key:
            self.set_persona(key)

    def set_persona(self, key):
        """Переключает учителя: имя, город, голос, промпт и новый диалог."""
        if key not in PERSONAS or key == self.persona_key:
            return
        self.persona_key = key
        self.persona = PERSONAS[key]
        save_persona_key(key)
        self.name_label.config(text=self.persona["name"])
        self.sub_label.config(text=f"English tutor  ·  {self.persona['home']}")
        self.update_avatar()
        self.root.title(f"{self.persona['name']} · English tutor")
        self.persona_box.set(f"{self.persona['name']} · {self.persona['voice']} голос")
        dbg("persona ->", key)
        self.interrupt("смена учителя")
        self.messages.clear()
        self.turn = 0
        self._pending = []
        self.drill = None
        self.barge_ready = False        # во время приветствия нового учителя не перебиваем
        self.text.delete("1.0", "end")
        if self.speaker:
            self.speaker.set_voice(self.persona["piper"], self.persona["kokoro"])
        if self.mode == "tenses":
            self.start_tense_mode()
        elif self.mode == "words":
            self.start_word_mode()
        elif self.vocab:
            threading.Thread(target=self.regreet, daemon=True).start()

    def regreet(self):
        """Приветствие от нового учителя (после смены личности)."""
        try:
            self.messages.clear()
            self.set_state("speaking")
            greeting = self.make_greeting()
            self.messages += [{"role": "user", "content": "[The learner has just opened the app.]"},
                              {"role": "assistant", "content": greeting}]
            self._tutor_started = False
            self._user_ru_shown = True
            self._question_seen = False
            self._skip_next_ru = False
            reader = ReplyReader(has_learner_line=False)
            gen = self.speaker.gen
            self.show_events(reader.feed(greeting) + reader.flush(), gen)
            if self._pending:
                pending, self._pending = self._pending, []
                threading.Thread(target=self.translate_missing, args=(pending,), daemon=True).start()
            self.say_text("\n")
            self.speaker.done(gen)
        except Exception as e:
            self.status(f"Ошибка: {e}")

    # ---- кнопка «держу, пока говорю» ----
    def start_hold(self, _event=None):
        """Пока кнопка нажата, пауза в речи не считается концом фразы."""
        if self.state == "loading":
            return "break"
        if not self.enabled:
            self.toggle()
        if self.state in ("thinking", "speaking"):
            self.interrupt("кнопка Говорить")
        self.mute_until = 0.0
        self.hold_release = False
        self.hold = True
        dbg("hold start")
        return "break"

    def stop_hold(self, _event=None):
        if not self.hold:
            return "break"
        self.hold = False
        self.hold_release = True
        dbg("hold stop")
        return "break"

    # ---- недостающие переводы ----
    def new_mark(self):
        """Метка сразу после только что вставленной строки: туда можно дописать перевод."""
        self._mark_n += 1
        name = f"after_{self._mark_n}"
        self.ui.put(("mark", name))
        return name

    def translate_missing(self, items):
        """Переводит строки, которым модель не дала перевод (одним запросом)."""
        try:
            lines = [text for _mark, text in items]
            prompt = ("Translate every line into natural Russian. Keep the same order, one "
                      "translation per line, no numbering, no quotes, no explanations:\n\n"
                      + "\n".join(lines))
            r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}],
                            stream=False, keep_alive=KEEP_ALIVE,
                            options=dict(OPTIONS, temperature=0.2))
            answer = (g(g(r, "message"), "content") or "").strip()
            got = [ln.strip() for ln in answer.splitlines() if ln.strip()]
            for (mark, _text), translation in zip(items, got):
                translation = translation.lstrip("0123456789.-) ").strip()
                self.ui.put(("insert_at", mark, parenthesise(translation) + "\n", "ru"))
            if len(got) < len(items):
                dbg("translate: не хватило строк", len(got), "из", len(items))
        except Exception as e:
            dbg("translate ERROR:", type(e).__name__, e)

    # ---- запуск ----
    def setup(self):
        try:
            self.set_state("loading")
            ensure_ollama(self.status)
            self.status("Загружаю Whisper...")
            self.whisper = load_whisper()
            self.status(f"Whisper: {WHISPER_DEVICE}")
            self.status("Загружаю Silero VAD...")
            self.vad_model = load_silero_vad()
            self.status(f"Прогреваю {OLLAMA_MODEL}...")
            ollama.generate(model=OLLAMA_MODEL, prompt="", keep_alive=KEEP_ALIVE, options=OPTIONS)
            self.vocab = Vocabulary()
            self.vocab_block = self.vocab.prompt_block()
            self.speaker = Speaker(self.on_speech_done, self.status,
                                   self.persona["piper"], self.persona["kokoro"])
            self.ui.put(("tense_ready",))
            threading.Thread(target=self.worker, daemon=True).start()
            threading.Thread(target=self.vad_loop, daemon=True).start()
            self.mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                      blocksize=BLOCK, callback=self.audio_cb)
            self.mic.start()
            self.status("Готовлю приветствие...")
            greeting = self.make_greeting()
            dbg("=== greeting:", greeting.replace("\n", " | ")[:220])
            self.messages += [{"role": "user", "content": "[The learner has just opened the app.]"},
                              {"role": "assistant", "content": greeting}]
            self.turn = 0
            self.ui.put(("focus", focus_label(self.turn)))
            self._tutor_started = False
            self._user_ru_shown = True      # переводить нечего: вы ещё не говорили
            self._question_seen = False
            self._skip_next_ru = False
            reader = ReplyReader(has_learner_line=False)
            self.set_state("speaking")
            gen = self.speaker.gen
            self.show_events(reader.feed(greeting) + reader.flush(), gen)
            if self._pending:                  # приветствие без переводов - допереводим
                pending, self._pending = self._pending, []
                threading.Thread(target=self.translate_missing, args=(pending,), daemon=True).start()
            self.say_text("\n")
            self.speaker.done(gen)
        except Exception as e:
            self.status(f"Ошибка: {e}")

    # ---- кнопки ----
    def interrupt(self, reason="кнопка"):
        dbg("interrupt:", reason, "state", self.state)
        self.cancel.set()
        if self.speaker:
            self.speaker.interrupt()

    def manual_interrupt(self):
        if self.state in ("thinking", "speaking", "hearing"):
            self.interrupt("кнопка Прервать")
            self.set_state("listening" if self.enabled else "paused")

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.toggle_btn.config(text="Пауза")
            self.set_state("listening")
        else:
            self.interrupt("пауза")
            self.toggle_btn.config(text="Продолжить")
            self.set_state("paused")

    def reset(self):
        if self.mode == "tenses":
            self.start_tense_mode()
            return
        if self.mode == "words":
            self.start_word_mode()
            return
        self.interrupt("новый диалог")
        self.messages.clear()
        self.turn = 0
        self._pending = []
        self.drill = None
        self.ui.put(("focus", ""))
        self.focus_tip.hide()
        self.focus_tip.set("")
        self.text.delete("1.0", "end")
        if self.enabled and self.state != "loading":
            self.set_state("listening")

    # ---- переключение режимов ----
    def _update_mode_buttons(self):
        styles = {"chat": "ChatOn.TButton", "tenses": "TenseOn.TButton",
                  "words": "WordsOn.TButton", "lookup": "LookupOn.TButton"}
        for mode, btn in (("chat", self.chat_btn), ("tenses", self.tense_btn),
                          ("words", self.words_btn), ("lookup", self.lookup_btn)):
            btn.config(style=styles[mode] if self.mode == mode else "Mode.TButton")
        if self.mode == "words":
            self.word_tools.pack(fill="x", pady=(8, 0))
        else:
            self.word_tools.pack_forget()

    def set_mode(self, mode):
        if not self.speaker:
            self.status("Ещё загружаюсь...")
            return
        if mode == "lookup":
            if self.mode == "lookup":
                self.mode = self.base_mode
                self._update_mode_buttons()
                self.set_state("listening" if self.enabled else "paused")
                return
            self.base_mode = self.mode
            self.mode = "lookup"
            self._update_mode_buttons()
            self.focus_tip.hide()
            self.focus_tip.set("")
            self.status("Спросите по-русски, как сказать слово")
            if self.enabled:
                self.set_state("listening")
            return
        if self.mode == "lookup" and mode == self.base_mode:
            self.mode = mode
            self._update_mode_buttons()
            self.set_state("listening" if self.enabled else "paused")
            return
        if mode == self.mode:
            return
        self.mode = mode
        self._update_mode_buttons()
        self.focus_tip.hide()
        self.focus_tip.set("")
        if mode == "chat":
            self.reset()
        elif mode == "tenses":
            self.start_tense_mode()
        elif mode == "words":
            self.start_word_mode()

    def start_tense_mode(self):
        """Начинает тренажёр с первого времени."""
        self.interrupt("режим времён")
        self.cancel.clear()
        self.messages.clear()
        self.turn = 0
        self._pending = []
        self.drill = None
        self.tense_i = 0
        self.tense_drills = list(TENSE_DRILLS)
        self.tense_current = None
        self.tense_correct = 0
        self.text.delete("1.0", "end")
        if not self.speaker:
            return
        self.set_state("thinking")
        gen = self.speaker.gen
        self._tutor_started = False
        self._user_ru_shown = True
        self._tense_gen_token = getattr(self, "_tense_gen_token", 0) + 1
        token = self._tense_gen_token
        self._tense_loading = True

        def worker():
            self.status("Придумываю новые фразы...")
            try:
                fresh = generate_tense_drills()
            except Exception as e:
                dbg("tense drills ERROR:", type(e).__name__, e)
                fresh = []
            finally:
                self._tense_loading = False
            if self.cancel.is_set() or self.mode != "tenses" or token != self._tense_gen_token:
                self.speaker.done(gen)
                return
            if fresh:
                self.tense_drills = fresh
            self.set_state("thinking")
            self.ask_tense_question(gen)
            self.status("Готов — просто говорите")
            self.speaker.done(gen)

        threading.Thread(target=worker, daemon=True).start()

    def ask_tense_question(self, gen):
        """Задаёт русскую фразу для текущего времени."""
        if self.tense_i >= len(self.tense_drills):
            self.finish_tense_mode(gen)
            return
        self.tense_current = self.tense_drills[self.tense_i]
        tense, russian, _english, hint = self.tense_current
        short_hint = hint.split("—")[0].split(" - ")[0].strip() or hint
        self.ui.put(("focus", f"Времена · {tense}"))
        self.focus_tip.set(f"{tense}: {short_hint}")
        instruction = f"Next: {tense}."
        self.say_text(instruction + "\n", "en_jn")
        self.say_text("«" + russian + "»\n", "task")
        self.say_text("Подсказка: " + short_hint + "\n", "hint")
        if self.speak(gen, instruction) and self.state == "thinking":
            self.set_state("speaking")

    def finish_tense_mode(self, gen):
        """Все фразы текущей порции пройдены."""
        self.ui.put(("focus", "Времена · готово"))
        self.focus_tip.hide()
        self.focus_tip.set("")
        line = "Done!"
        self.say_text(line + "\n", "en_jn")
        if self.speak(gen, line) and self.state == "thinking":
            self.set_state("speaking")

    def process_tense(self, user_text):
        """Проверяет ответ ученика в тренажёре времён."""
        if getattr(self, "_tense_loading", False):
            self.status("Ещё готовлю фразы...")
            if self.enabled and self.state == "thinking":
                self.set_state("listening")
            return
        if self.tense_i >= len(self.tense_drills):
            self.status("Все фразы пройдены — нажмите «Новый диалог»")
            if self.enabled and self.state == "thinking":
                self.set_state("listening")
            return
        self.status("Проверяю...")
        gen = self.speaker.gen
        self._tutor_started = False
        self._pending = []
        self._last_fix = None
        tense, russian, english, _hint = (self.tense_current
                                          or self.tense_drills[self.tense_i])
        repeat = self.drill is not None
        answer = ask(tense_eval_prompt(self.persona, tense, russian, english, repeat)
                     + "\n\nLEARNER'S ANSWER: " + user_text, temperature=0.1)
        if self.cancel.is_set():
            if self.enabled and self.state == "thinking":
                self.set_state("listening")
            return
        if not answer.strip():
            answer = (f"Fix: {english}\n({russian})\n"
                      f"Почему: здесь нужна конструкция {tense}.\n"
                      "Say it again.")
        reader = ReplyReader(has_learner_line=False)
        events = reader.feed(answer) + reader.flush()
        drill_line = "Say it again."
        events = [e for e in events
                  if not (e[0] == "en" and re.match(r"(?:now )?say it", e[1].strip(), re.I))]
        fix = next((t for k, t in events if k == "fix"), None)
        if fix:
            self._question_seen = False
            self._skip_next_ru = False
            self.show_events(events, gen)
            self.say_text(drill_line + "\n", "en_jn")
            if self.speak(gen, drill_line) and self.state == "thinking":
                self.set_state("speaking")
            self.drill = fix
            pending, self._pending = self._pending, []
            if pending:
                threading.Thread(target=self.translate_missing, args=(pending,), daemon=True).start()
            self.status("Готов — повторите фразу")
        else:
            self.drill = None
            self.tense_i += 1
            self.tense_correct = getattr(self, "tense_correct", 0) + 1
            if self.tense_correct % 3 == 0:
                praise_en, _praise_ru = random.choice(PRAISE)
                self.say_text(praise_en + "\n", "en_jn")
                if self.speak(gen, praise_en) and self.state == "thinking":
                    self.set_state("speaking")
            if self.tense_i >= len(self.tense_drills):
                self.finish_tense_mode(gen)
            else:
                self.ask_tense_question(gen)
            self.status("Готов — просто говорите")
        self.speaker.done(gen)

    # ---- режим «Изучение слов» ----
    SKIP_PHRASES = (
        "следующее слово", "следующее", "другое слово", "пропустить", "пропусти",
        "дальше", "next word", "another word", "skip", "move on",
        "sleduyush", "sleduyus", "propust", "dal'she", "dalshe", "drugoye",
    )
    HINT_PHRASES = (
        "подсказка", "подскажи", "подскажите", "не знаю", "hint",
        "i don't know", "i dont know", "dunno",
        "podskazka", "podskazhi", "ne znayu",
    )

    @staticmethod
    def _norm_cmd(text):
        t = text.lower().strip()
        t = re.sub(r"[^\w\s']", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _is_skip_command(self, text):
        t = self._norm_cmd(text)
        if not t:
            return False
        target = (self.word_current or {}).get("word", "").lower().strip()
        if target and t == self._norm_cmd(target):
            return False
        return any(p in t for p in self.SKIP_PHRASES)

    def _is_hint_command(self, text):
        t = self._norm_cmd(text)
        if not t:
            return False
        target = (self.word_current or {}).get("word", "").lower().strip()
        if target and t == self._norm_cmd(target):
            return False
        return any(p in t for p in self.HINT_PHRASES)

    @staticmethod
    def _word_hint_text(english, level):
        letters = [c for c in english if c.isalpha()]
        first = letters[0].lower() if letters else "?"
        first_two = "".join(letters[:2]).lower() if len(letters) >= 2 else first
        if level <= 1:
            return f'"{first}..."'
        if level == 2:
            return f'"{first_two}..."'
        return f'"{english}"'

    def _show_word_hint(self, gen):
        if not self.word_current:
            return
        english = self.word_current.get("word", "")
        if not english:
            return
        hint = self._word_hint_text(english, max(1, self.word_hint_level))
        self.say_text(hint + "\n", "hint")
        if self.speak(gen, hint) and self.state == "thinking":
            self.set_state("speaking")

    def word_hint(self):
        """Кнопка «Подсказка»: первая буква, потом две, потом всё слово."""
        if not self.speaker or not self.word_current:
            return
        gen = self.speaker.gen
        self.word_hint_level = min(self.word_hint_level + 1, 3)
        self._show_word_hint(gen)
        self.speaker.done(gen)

    def word_skip(self):
        """Кнопка «Следующее слово»: пропустить текущее слово."""
        if not self.speaker:
            return
        gen = self.speaker.gen
        self._word_skip(gen)
        self.speaker.done(gen)

    def add_words(self):
        """Кнопка «Добавить слова»: берёт слова из нового диалога."""
        if not self.vocab or not self.speaker:
            return
        if getattr(self, "_word_gen_busy", False):
            self.status("Уже подбираю слова...")
            return
        gen = self.speaker.gen
        self.status("Придумываю новый диалог и слова...")
        self._word_gen_token = getattr(self, "_word_gen_token", 0) + 1
        token = self._word_gen_token

        def worker():
            have = sum(1 for w in self.vocab.words if w.get("translation"))
            added = self.ensure_word_supply(have + 5)
            if self.mode != "words" or token != self._word_gen_token:
                self.speaker.done(gen)
                return
            if added:
                line = "New words!"
            else:
                line = "No words left."
            self.say_text(line + "\n", "en_jn")
            if self.speak(gen, line) and self.state == "thinking":
                self.set_state("speaking")
            self.ask_word_question(gen)
            self.speaker.done(gen)

        threading.Thread(target=worker, daemon=True).start()

    def _word_skip(self, gen, reveal=False):
        """Пропускает текущее слово и задаёт следующее."""
        if not self.word_current:
            return
        item = self.word_current
        english = item.get("word", "")
        if english and self.vocab:
            self.vocab.skip(english)
            self.vocab_block = self.vocab.prompt_block()
        self.word_current = None
        self.word_drill = None
        self.word_attempts = 0
        self.word_hint_level = 0
        if reveal and english:
            line = f'"{english}".'
        else:
            line = "Next."
        self.say_text(line + "\n", "en_jn")
        if self.speak(gen, line) and self.state == "thinking":
            self.set_state("speaking")
        self.ask_word_question(gen)
        self.status("Готов — просто говорите")

    def generate_words_from_dialogue(self, count=5):
        """Просит модель придумать новый диалог и взять из него слова."""
        topic = random.choice(TOPICS)
        scenario = random.choice(SCENARIOS)
        prompt = (
            f"You are {self.persona['name']}, an English teacher. "
            f"Invent a short, fresh dialogue (6-8 lines) between you and the learner "
            f"on the topic \"{topic}\" in the situation \"{scenario}\". "
            "This dialogue has not happened yet.\n"
            f"After the dialogue, write a line \"WORDS:\" and then exactly {count} lines, each in the format:\n"
            "english word or phrase — русский перевод\n"
            "Use only useful everyday words or phrases from the dialogue. "
            "No numbering, no extra comments."
        )
        try:
            text = ask(prompt, temperature=0.8)
        except Exception as e:
            dbg("generate_words_from_dialogue:", e)
            return []
        return parse_word_pairs(text)[:count]

    def ensure_word_supply(self, min_count=5):
        """Добирает слова: сначала из нового диалога, потом из банка."""
        if not self.vocab:
            return 0
        have = sum(1 for w in self.vocab.words if w.get("translation"))
        if have >= min_count:
            return 0
        if getattr(self, "_word_gen_busy", False):
            return 0
        self._word_gen_busy = True
        try:
            need = min_count - have
            self.status("Придумываю новый диалог и слова...")
            pairs = self.generate_words_from_dialogue(need)
            existing = {w.get("word", "").lower() for w in self.vocab.words}
            added = 0
            for word, translation in pairs:
                if word.lower() in existing:
                    continue
                if self.vocab.add(f"{word} — {translation}"):
                    existing.add(word.lower())
                    added += 1
            if added < need:
                added += self.vocab.ensure_min_words(min_count)
            if added:
                self.vocab_block = self.vocab.prompt_block()
            return added
        finally:
            self._word_gen_busy = False

    def _add_words_bg(self):
        """Тихо добирает новую порцию слов в фоне."""
        if not self.vocab:
            return
        have = sum(1 for w in self.vocab.words if w.get("translation"))
        self.ensure_word_supply(have + 5)
        if self.mode == "words":
            self.status("Готов — просто говорите")

    def start_word_mode(self):
        """Начинает тренажёр слов по словарю."""
        self.interrupt("режим слов")
        self.cancel.clear()
        self.messages.clear()
        self.turn = 0
        self._pending = []
        self.drill = None
        self.word_drill = None
        self.word_current = None
        self.last_word = None
        self.word_attempts = 0
        self.word_hint_level = 0
        self.word_correct = 0
        self._word_session_seen = set()
        self.text.delete("1.0", "end")
        if not self.speaker:
            return
        self.set_state("thinking")
        gen = self.speaker.gen
        self._tutor_started = False
        self._user_ru_shown = True
        self._word_gen_token = getattr(self, "_word_gen_token", 0) + 1
        token = self._word_gen_token

        def worker():
            added = self.ensure_word_supply(5)
            if self.mode != "words" or token != self._word_gen_token:
                self.speaker.done(gen)
                return
            if not self.vocab or not self.vocab.words:
                line = "No words yet."
                self.say_text(line + "\n", "en_jn")
                if self.speak(gen, line) and self.state == "thinking":
                    self.set_state("speaking")
            else:
                if added:
                    line = "New words!"
                    self.say_text(line + "\n", "en_jn")
                    if self.speak(gen, line) and self.state == "thinking":
                        self.set_state("speaking")
                self.ask_word_question(gen)
            self.speaker.done(gen)

        threading.Thread(target=worker, daemon=True).start()

    def ask_word_question(self, gen):
        """Задаёт русское слово/фразу, ученик отвечает по-английски."""
        if not self.vocab:
            return
        if not hasattr(self, "_word_session_seen"):
            self._word_session_seen = set()
        seen = self._word_session_seen
        unseen = [w for w in self.vocab.words
                  if w.get("translation") and w.get("word") not in seen]
        if len(unseen) < 2 and not getattr(self, "_word_gen_busy", False):
            threading.Thread(target=self._add_words_bg, daemon=True).start()
        if not unseen:
            # новых слов нет - ждём фоновую порцию и не повторяем старые
            item = None
            for _ in range(150):          # ~30 секунд
                if self.cancel.is_set() or self.mode != "words":
                    return
                time.sleep(0.2)
                if getattr(self, "_word_gen_busy", False):
                    continue
                fresh = [w for w in self.vocab.words
                         if w.get("translation") and w.get("word") not in seen]
                if fresh:
                    item = self.vocab.next_due(exclude=self.last_word, seen=seen)
                break
        else:
            item = self.vocab.next_due(exclude=self.last_word, seen=seen)
        if not item:
            self.finish_word_mode(gen)
            return
        self.word_current = item
        self.last_word = item.get("word")
        seen.add(self.last_word)
        self.word_attempts = 0
        self.word_hint_level = 0
        russian = item.get("translation") or item.get("word", "")
        self.ui.put(("focus", "Слова · повторение"))
        self.focus_tip.set(russian)
        self.say_text("«" + russian + "»\n", "task")
        self.speaker.say(gen, russian, "ru")
        if self.state == "thinking":
            self.set_state("speaking")

    def finish_word_mode(self, gen):
        """Словарь пуст или всё повторено."""
        self.word_current = None
        self.word_drill = None
        self.ui.put(("focus", "Слова · готово"))
        self.focus_tip.hide()
        self.focus_tip.set("")
        line = "Done!"
        self.say_text(line + "\n", "en_jn")
        if self.speak(gen, line) and self.state == "thinking":
            self.set_state("speaking")

    def process_word(self, user_text):
        """Проверяет ответ ученика в тренажёре слов."""
        if not self.word_current:
            self.ask_word_question(self.speaker.gen)
            return
        gen = self.speaker.gen
        text = user_text.strip()
        if self._is_skip_command(text):
            self._word_skip(gen)
            self.speaker.done(gen)
            return
        if self._is_hint_command(text):
            self.word_hint_level = min(self.word_hint_level + 1, 3)
            self._show_word_hint(gen)
            self.speaker.done(gen)
            return
        self.status("Проверяю...")
        self._tutor_started = False
        self._pending = []
        self._last_fix = None
        item = self.word_current
        english = item.get("word", "")
        russian = item.get("translation") or english
        repeat = self.word_drill is not None
        answer = ask(word_eval_prompt(self.persona, russian, english, repeat)
                     + "\n\nLEARNER'S ANSWER: " + user_text, temperature=0.1)
        if self.cancel.is_set():
            if self.enabled and self.state == "thinking":
                self.set_state("listening")
            return
        if not answer.strip():
            answer = (f"Fix: {english}\n({russian})\n"
                      "Почему: это слово из твоего словаря.\n"
                      "Say it again.")
        reader = ReplyReader(has_learner_line=False)
        events = reader.feed(answer) + reader.flush()
        drill_line = "Say it again."
        events = [e for e in events
                  if not (e[0] == "en" and re.match(r"(?:now )?say it", e[1].strip(), re.I))]
        fix = next((t for k, t in events if k == "fix"), None)
        if fix:
            self._question_seen = False
            self._skip_next_ru = False
            self.show_events(events, gen)
            self.say_text(drill_line + "\n", "en_jn")
            if self.speak(gen, drill_line) and self.state == "thinking":
                self.set_state("speaking")
            self.word_drill = english
            self.vocab.record(english, False)
            self.vocab_block = self.vocab.prompt_block()
            pending, self._pending = self._pending, []
            if pending:
                threading.Thread(target=self.translate_missing, args=(pending,), daemon=True).start()
            self.word_attempts += 1
            if self.word_attempts >= 3:
                self._word_skip(gen, reveal=True)
                self.speaker.done(gen)
                return
            self.word_hint_level = max(self.word_hint_level,
                                       1 if self.word_attempts == 1 else 2)
            self._show_word_hint(gen)
            self.status("Готов — повторите слово")
        else:
            self.word_drill = None
            self.word_attempts = 0
            self.word_hint_level = 0
            self.vocab.record(english, True)
            self.vocab_block = self.vocab.prompt_block()
            self.word_correct = getattr(self, "word_correct", 0) + 1
            if self.word_correct % 3 == 0:
                praise_en, _praise_ru = random.choice(PRAISE)
                self.say_text(praise_en + "\n", "en_jn")
                if self.speak(gen, praise_en) and self.state == "thinking":
                    self.set_state("speaking")
            self.ask_word_question(gen)
            self.status("Готов — просто говорите")
        self.speaker.done(gen)

    def process_lookup(self, audio):
        """Русский вопрос -> английское слово; разговор не сдвигается."""
        self.status("Слушаю по-русски...")
        segs, _ = self.whisper.transcribe(audio, language="ru", vad_filter=True)
        question = " ".join(s.text.strip() for s in segs).strip()
        if len(question) < 2 or not re.search(r"\w", question):
            self.status("Не расслышал")
            if self.enabled:
                self.set_state("listening")
            return
        self.say_text("Вы\n", "name_me")
        self.say_text(question + "\n", "en_me")
        self.status("Перевожу...")
        prompt = ("The learner is asking in Russian how to say something in English. "
                  "Their question: «" + question + "». "
                  "Reply with exactly two lines and nothing else:\n"
                  "<the English word or phrase>\n"
                  "(<its Russian translation>)\n"
                  "Use the most common everyday English word or phrase.")
        answer = ask(prompt, temperature=0.1)
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        if not lines:
            self.status("Не понял вопрос")
            if self.enabled:
                self.set_state("listening")
            return
        english = lines[0].strip().strip('"').strip("«»").strip()
        english = re.sub(r"^(?:english|the english word(?: or phrase)? is)\\s*[:\\-]?\\s*",
                         "", english, flags=re.I).strip()
        russian = lines[1].strip() if len(lines) > 1 else question
        russian = russian.strip("()").strip()
        if not english:
            self.status("Не понял вопрос")
            if self.enabled:
                self.set_state("listening")
            return
        gen = self.speaker.gen
        self._tutor_started = False
        self.start_tutor_block()
        self.say_text(english + "\n", "en_jn")
        self.say_text(parenthesise(russian) + "\n", "ru")
        if self.speak(gen, english) and self.state == "thinking":
            self.set_state("speaking")
        self.speaker.done(gen)

    def make_greeting(self):
        # Простое приветствие: не ждём модель и не придумываем лишнего.
        return default_greeting(self.persona)

    def wipe_vocab(self):
        if not self.vocab or not messagebox.askyesno("Стереть словарь",
                                                     "Удалить все сохранённые слова?"):
            return

        def run():
            self.vocab.wipe()
            self.vocab_block = self.vocab.prompt_block()
            self.status("Словарь стёрт")
        threading.Thread(target=run, daemon=True).start()

    # ---- микрофон -> VAD ----
    def audio_cb(self, indata, frames, t, status):
        self.audio_q.put(indata[:, 0].copy())

    def vad_loop(self):
        vad = VADIterator(self.vad_model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLE_RATE,
                          min_silence_duration_ms=SILENCE_MS, speech_pad_ms=100)
        preroll = collections.deque(maxlen=PREROLL_BLOCKS)
        utter, speaking, interrupted, loud = [], False, False, 0
        while True:
            block = self.audio_q.get()
            level = float(np.sqrt(np.mean(block ** 2)))

            # --- режим «держу кнопку»: VAD не решает, когда фраза закончена ---
            if self.hold and self.enabled and self.state != "loading":
                if not speaking:
                    speaking, interrupted, utter, loud = True, False, list(preroll), 0
                    vad.reset_states()
                    dbg("hold start, state", self.state)
                    if self.state in ("listening", "hearing"):
                        self.set_state("hearing")
                utter.append(block)
                continue

            if self.hold_release:
                self.hold_release = False
                if speaking:
                    audio = np.concatenate(utter)
                    utter, speaking, loud = [], False, 0
                    vad.reset_states()
                    dbg("hold -> фраза", round(len(audio) / SAMPLE_RATE, 2), "с")
                    if len(audio) / SAMPLE_RATE >= MIN_UTTER_S:
                        self.set_state("thinking")
                        self.utt_q.put(audio)
                    elif self.state == "hearing":
                        self.set_state("listening")
                continue

            busy = self.state in ("thinking", "speaking") or time.time() < self.mute_until
            if not self.enabled or self.state == "loading" or (busy and not self.barge_flag):
                # пауза, загрузка или «эхо-защита»: звук игнорируем и сбрасываем состояние VAD
                utter, speaking, interrupted, loud = [], False, False, 0
                preroll.clear()
                vad.reset_states()
                continue

            ev = vad(torch.from_numpy(block))
            if ev and "start" in ev and not speaking:
                speaking, interrupted, utter, loud = True, False, list(preroll), 0
                dbg("vad start  state", self.state, "level", round(level, 4))
                if self.state == "listening":
                    self.set_state("hearing")
            if speaking:
                utter.append(block)
                if level > BARGE_MIN_RMS:
                    loud += 1
                # перебивание: человек говорит достаточно долго и достаточно громко
                if (busy and not interrupted and self.barge_ready
                        and loud * BLOCK / SAMPLE_RATE >= BARGE_MIN_S):
                    interrupted = True
                    dbg("barge-in  state", self.state, "loud", round(loud * BLOCK / SAMPLE_RATE, 2), "с")
                    self.interrupt("голос")
                    self.set_state("hearing")
            else:
                preroll.append(block)

            if ev and "end" in ev and speaking:
                audio = np.concatenate(utter)
                utter, speaking, loud = [], False, 0
                if len(audio) / SAMPLE_RATE >= MIN_UTTER_S:
                    self.set_state("thinking")
                    self.utt_q.put(audio)
                elif self.state == "hearing":
                    self.set_state("listening")

    # ---- обработка фраз (по одной, строго по очереди) ----
    def worker(self):
        while True:
            audio = self.utt_q.get()
            self.cancel.clear()
            try:
                self.process(audio)
            except Exception as e:
                self.status(f"Ошибка: {e}")
                if self.enabled:
                    self.set_state("listening")

    def on_speech_done(self, gen):
        if gen == self.speaker.gen and self.state in ("thinking", "speaking") and self.enabled:
            self.barge_ready = True      # приветствие доиграло - теперь перебивать можно
            self.mute_until = time.time() + ECHO_TAIL_S
            self.set_state("listening")

    # ---- вывод реплики Джона: английский + перевод ----
    def start_tutor_block(self):
        """Печатает имя Джона один раз за реплику."""
        if self._tutor_started:
            return
        self._tutor_started = True
        self.say_text(f"{self.persona['name']}\n", "name_jn")

    def speak(self, gen, s):
        """Ставит английскую фразу в озвучку; True, если фраза ушла в очередь."""
        s = MD_RE.sub("", s)               # никаких звёздочек и подчёркиваний вслух
        s = PAREN_RE.sub(" ", s).strip()   # переводы в скобках вслух не читаем
        if not s or CYR.search(s):
            return False
        self.speaker.say(gen, s)
        return True

    def show_events(self, events, gen):
        """Показывает разобранные строки ответа и озвучивает только английский."""
        for kind, text in events:
            if kind == "user_ru":
                if not self._tutor_started:
                    self.say_text(parenthesise(text) + "\n", "ru")
                    self._user_ru_shown = True
                    continue
                kind = "ru"   # перевод уже начатой реплики Джона
            if kind == "en":
                is_question = "?" in text
                if is_question and self._question_seen:
                    self._skip_next_ru = True
                    continue
                if is_question:
                    self._question_seen = True
                self._skip_next_ru = False
                self.start_tutor_block()
                self.say_text(text + "\n", "en_jn")
                self._pending.append((self.new_mark(), text))   # ждём перевод этой строки
                for sentence in drain(text, final=True)[0]:
                    if self.speak(gen, sentence) and self.state == "thinking":
                        self.set_state("speaking")
            elif kind == "ru":
                if self._skip_next_ru:
                    self._skip_next_ru = False
                    continue
                self.say_text(parenthesise(text) + "\n", "ru")
                if self._pending:
                    self._pending.pop()          # перевод пришёл - строка закрыта
            elif kind == "fix":
                self.start_tutor_block()
                self._last_fix = text
                self.say_text("Fix: " + text + "\n", "fix")
                self._pending.append((self.new_mark(), text))
                if self.speak(gen, "Better: " + text.split("(")[0]) and self.state == "thinking":
                    self.set_state("speaking")
            elif kind == "note":
                self.say_text("Почему: " + text + "\n", "note")
            elif kind == "word":
                self.say_text("Слово: " + text + "\n", "word")
                if self.vocab:
                    self.vocab.add(text)
                    self.vocab_block = self.vocab.prompt_block()
                english = re.split(r"\s+[—–-]\s+|\s*:\s*", text, maxsplit=1)[0].strip()
                if english and self.speak(gen, english) and self.state == "thinking":
                    self.set_state("speaking")

    def process(self, audio):
        if self.mode == "lookup":
            self.process_lookup(audio)
            return
        self.status("Распознаю...")
        segs, _ = self.whisper.transcribe(audio, language=WHISPER_LANGUAGE, vad_filter=True)
        user_text = " ".join(s.text.strip() for s in segs).strip()
        if len(user_text) < 2 or not re.search(r"\w", user_text):
            self.status("Не расслышал")
            if self.enabled and self.state == "thinking":
                self.set_state("listening")
            return
        self.turn += 1
        self.say_text("Вы\n", "name_me")
        self.say_text(user_text + "\n", "en_me")
        if self.mode == "tenses":
            self.process_tense(user_text)
            return
        if self.mode == "words":
            self.process_word(user_text)
            return
        self.messages.append({"role": "user", "content": user_text})

        self.ui.put(("focus", focus_label(self.turn)))
        self.status("Думаю...")
        gen = self.speaker.gen
        reader = ReplyReader()
        self._tutor_started = False
        self._user_ru_shown = False
        self._pending = []
        self._last_fix = None
        self._question_seen = False
        self._skip_next_ru = False
        drill = self.drill
        self.drill = None
        full = ""
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": system_prompt(self.persona) + self.vocab_block
                       + focus_block(self.turn) + drill_block(drill)}]
            + self.messages[-HISTORY_TURNS:],
            stream=True, keep_alive=KEEP_ALIVE, options=OPTIONS,
        )
        try:
            for chunk in stream:
                if self.cancel.is_set():
                    break
                tok = g(g(chunk, "message"), "content") or ""
                full += tok
                self.show_events(reader.feed(tok), gen)
        finally:
            stream.close()  # закрывает соединение с Ollama - генерация останавливается

        cancelled = self.cancel.is_set()
        if not cancelled:
            self.show_events(reader.flush(), gen)
        pending, self._pending = self._pending, []
        if pending and not cancelled:      # строки без перевода - допереводим сами
            threading.Thread(target=self.translate_missing, args=(pending,), daemon=True).start()
        if cancelled:
            self.say_text("⏹ прервано\n", "stop")
        if full.strip():
            self.messages.append({"role": "assistant", "content": full.strip()})
        if not cancelled:
            # была ошибка - разговор ждёт, пока ученик не повторит исправленную фразу
            self.drill = self._last_fix
            self.status("Готов — просто говорите")
            self.speaker.done(gen)  # когда озвучка закончится, придёт on_speech_done
        else:
            self.drill = drill      # прервали - повтор всё ещё ждём


if __name__ == "__main__":
    enable_hi_dpi()          # до создания окна: чёткие шрифты при масштабировании Windows
    root = tk.Tk()
    try:                     # иконка «e» в заголовке и на панели задач
        root.iconbitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "english_tutor.ico"))
    except Exception:
        pass
    App(root)
    root.mainloop()
