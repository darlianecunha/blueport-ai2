# -*- coding: utf-8 -*-
"""
BluePort AI - Telegram bot for waste classification using CLIP vision.
Sends a photo and receives the waste type prediction with confidence score.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from waste_vision import predict_path, get_stats, get_count

# ------------ Config ------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
log = logging.getLogger("blueport-bot")


# ------------ Handlers ------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🤖 <b>BluePort AI</b> — classificação de resíduos por imagem.\n\n"
        "Envie uma <b>foto</b> e eu direi a classe e a confiança.\n\n"
        "Comandos:\n"
        "• /stats — total analisado + confiança média\n"
        "• /count — número total de imagens processadas\n"
    )
    await update.message.reply_html(text)


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        s = get_stats()
        await update.message.reply_html(
            f"📊 <b>Estatísticas</b>\n"
            f"Total no log: <b>{s['total']}</b>\n"
            f"Confiança média: <b>{s['media_confianca']:.0%}</b>"
        )
    except Exception as e:
        log.exception("Erro em /stats")
        await update.message.reply_text(f"⚠️ Erro ao ler estatísticas: {e}")


async def count_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.reply_html(f"🧮 Imagens analisadas: <b>{get_count()}</b>")
    except Exception as e:
        log.exception("Erro em /count")
        await update.message.reply_text(f"⚠️ Erro ao ler contador: {e}")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        img_path = TMP_DIR / f"{photo.file_id}.jpg"
        await file.download_to_drive(custom_path=str(img_path))

        res = predict_path(str(img_path))
        label = res["label"]
        conf = res["confidence"]
        count_total = res["count_total"]

        reply = (
            "♻️ <b>Classe:</b> {label}\n"
            "🔎 <b>Confiança:</b> {conf:.0%}\n"
            "📸 <b>Imagens analisadas:</b> {count}"
        ).format(label=label, conf=conf, count=count_total)

        await update.message.reply_html(reply)

    except Exception as e:
        log.exception("Erro ao processar foto")
        await update.message.reply_text(f"⚠️ Não consegui analisar a imagem.\nDetalhes: {e}")


# ------------ Main ------------
def main() -> None:
    if not TOKEN:
        raise RuntimeError(
            "Token não encontrado. Defina TELEGRAM_BOT_TOKEN no arquivo .env"
        )

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler(["start", "help"], start_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("count", count_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    log.info("BluePort AI bot rodando…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
