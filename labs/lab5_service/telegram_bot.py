import asyncio
import logging
import os
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from telegram import InputFile, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .tts_pipeline import SAMPLE_RATE, TTSPipeline

VOICE_RATE = 24000
RUN_DIR = Path(__file__).resolve().parents[1] / "lab4_acoustic" / "saved_runs"


def split_text(text: str, limit: int = 50):
    words = text.split()
    while len(words) > limit:
        cut = next(
            (i for i in range(limit, limit // 2, -1)
             if words[i - 1].endswith((",", ";", ":", ".", "!", "?"))),
            limit,
        )
        yield " ".join(words[:cut])
        words = words[cut:]
    if words:
        yield " ".join(words)


def make_voice(pipeline: TTSPipeline, text: str) -> bytes:
    parts = []
    for chunk in split_text(text):
        wav, _ = pipeline(chunk)
        if not wav.size or not np.isfinite(wav).all():
            raise ValueError("Пустой или некорректный сигнал")
        parts.append(wav)

    if not parts:
        raise ValueError("Нет текста для синтеза")

    wav = resample_poly(np.concatenate(parts), up=VOICE_RATE, down=SAMPLE_RATE)
    buffer = BytesIO()
    sf.write(buffer, wav, VOICE_RATE, format="OGG", subtype="OPUS")
    return buffer.getvalue()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is not None:
        await message.reply_text("Привет! Пришли текст на русском, и я отвечу голосовым.")


async def speak(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return

    text = message.text.strip()
    if re.search(r"[A-Za-z]|\d", text):
        await message.reply_text(
            "Напиши числа словами, а иностранные слова кириллицей."
        )
        return

    try:
        voice = await asyncio.to_thread(
            make_voice, context.bot_data["pipeline"], text
        )
    except ValueError:
        await message.reply_text("Не удалось озвучить текст. Попробуй переформулировать.")
        return
    except (RuntimeError, OSError):
        logging.exception("Ошибка при синтезе голосового")
        await message.reply_text("Не удалось создать голосовое. Попробуй ещё раз.")
        return

    await message.reply_voice(voice=InputFile(voice, filename="speech.ogg"))


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Сначала TELEGRAM_BOT_TOKEN в терминале.")

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(15)
        .read_timeout(30)
        .get_updates_connect_timeout(15)
        .get_updates_read_timeout(30)
        .media_write_timeout(60)
        .build()
    )
    app.bot_data["pipeline"] = TTSPipeline(run_dir=RUN_DIR, device="cpu")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, speak))

    print("Модели загружены.", flush=True)
    app.run_polling()


if __name__ == "__main__":
    main()