# -*- coding: utf-8 -*-
"""Генерация аватаров-иллюстраций (временный файл, dev-time)."""
from PIL import Image, ImageDraw

S = 512          # рисуем крупно, потом уменьшаем - сглаживание
OUT = 256


def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def draw(palette, hair, shirt, skin, female=False):
    top, bottom = (rgb(c) for c in palette)
    hair, shirt, skin = rgb(hair), rgb(shirt), rgb(skin)
    shadow = tuple(max(0, c - 40) for c in skin)

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    grad = Image.new("RGB", (1, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / (S - 1)
        gd.point((0, y), tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)))
    grad = grad.resize((S, S))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=255)
    img.paste(grad, (0, 0), mask)

    def e(x0, y0, x1, y1, fill):
        d.ellipse((x0 * S, y0 * S, x1 * S, y1 * S), fill=fill)

    def r(x0, y0, x1, y1, fill, radius=0.02):
        d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), radius=radius * S, fill=fill)

    if female:                        # длинные волосы за головой
        e(0.20, 0.30, 0.80, 0.98, hair)

    r(0.455, 0.60, 0.545, 0.82, shadow)   # шея
    e(0.08, 0.76, 0.92, 1.24, shirt)      # плечи
    e(0.305, 0.235, 0.695, 0.735, skin)
    e(0.285, 0.455, 0.325, 0.525, skin)
    e(0.675, 0.455, 0.715, 0.525, skin)

    if female:
        d.pieslice((0.29 * S, 0.20 * S, 0.71 * S, 0.62 * S), 180, 360, fill=hair)
        r(0.255, 0.36, 0.325, 0.90, hair, 0.05)
        r(0.675, 0.36, 0.745, 0.90, hair, 0.05)
    else:
        d.pieslice((0.295 * S, 0.195 * S, 0.705 * S, 0.60 * S), 180, 360, fill=hair)
        e(0.30, 0.36, 0.355, 0.47, hair)

    ink = (58, 45, 40)
    for cx in (0.442, 0.558):
        e(cx - 0.019, 0.492, cx + 0.019, 0.531, ink)
    d.arc((0.405 * S, 0.425 * S, 0.475 * S, 0.475 * S), 200, 340, fill=hair, width=int(0.016 * S))
    d.arc((0.525 * S, 0.425 * S, 0.595 * S, 0.475 * S), 200, 340, fill=hair, width=int(0.016 * S))
    d.arc((0.470 * S, 0.520 * S, 0.530 * S, 0.575 * S), 40, 140, fill=shadow, width=int(0.012 * S))
    d.arc((0.462 * S, 0.545 * S, 0.538 * S, 0.615 * S), 20, 160,
          fill=(178, 86, 78), width=int(0.018 * S))

    # обрезаем всё по кругу, чтобы плечи не вылезали за рамку
    img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", (S, S), 0), mask))
    return img.resize((OUT, OUT), Image.LANCZOS)


john = draw(("#123d2c", "#1f7a45"), hair="#33261f", shirt="#e2e8f0", skin="#e8b48f")
jane = draw(("#4c1d95", "#7c3aed"), hair="#6b3b25", shirt="#f7e6d8", skin="#f0c3a0", female=True)

john.save(r"D:\src\english_tutor\avatar_john.png")
jane.save(r"D:\src\english_tutor\avatar_jane.png")

preview = Image.new("RGBA", (OUT * 2 + 30, OUT + 20), (238, 242, 246, 255))
preview.paste(john, (10, 10), john)
preview.paste(jane, (OUT + 20, 10), jane)
preview.save(r"D:\src\english_tutor\_avatars_preview.png")
print("готово")
