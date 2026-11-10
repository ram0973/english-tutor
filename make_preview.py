"""Временный харнесс: поднимает окно приложения с демо-диалогом и снимает ui_preview.png.

Запускается Windows-питоном (там есть tkinter и все зависимости).
Аудио/Whisper/Ollama не поднимаются: App.setup подменён заглушкой.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import english_tutor_memory as et

et.enable_hi_dpi()
et.App.setup = lambda self: None          # без микрофона, Whisper и Ollama

root = et.tk.Tk()
try:
    root.iconbitmap(os.path.join(HERE, "english_tutor.ico"))
except Exception:
    pass
app = et.App(root)


def seed():
    s = app.say_text
    s("Jane\n", "name_jn")
    s("Hi! I'm Jane, your English tutor.\n", "en_jn")
    s("(Привет! Я Джейн, ваш преподаватель английского.)\n", "ru")
    s("How was your day?\n", "en_jn")
    s("(Как прошёл ваш день?)\n", "ru")

    s("Вы\n", "name_me")
    s("I goed to the park with my friend.\n", "en_me")
    s("(Я пошёл в парк с другом.)\n", "ru")

    s("Jane\n", "name_jn")
    s("Nice! But we say \"went\", not \"goed\".\n", "en_jn")
    s("Fix: I went to the park with my friend.\n", "fix")
    s("Почему: go — неправильный глагол, прошедшее время — went.\n", "note")
    s("(Отлично! Но мы говорим \"went\", а не \"goed\".)\n", "ru")
    s("Where did you go next?\n", "en_jn")
    s("(Куда вы пошли потом?)\n", "ru")

    app.ui.put(("status", "Готов — просто говорите"))
    app.ui.put(("focus", et.focus_label(0)))
    app.set_state("listening")
    for b in app.mode_btns:
        b.config(state="normal")
    app.chat_btn.config(style="ChatOn.TButton")


def capture():
    root.update_idletasks()
    root.update()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    time.sleep(0.8)
    root.update()

    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    from PIL import ImageGrab
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
    out = os.path.join(HERE, "ui_preview.png")
    img.save(out)
    print("saved", out, img.size)
    root.destroy()


root.after(400, seed)
root.after(1800, capture)
root.mainloop()
