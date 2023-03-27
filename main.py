from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InputFile
from aiogram.bot.api import TelegramAPIServer
from aiogram.utils.exceptions import ValidationError
import asyncio
from os import getenv
from sys import exit

BOT_TOKEN = getenv("BOT_TOKEN")
if not BOT_TOKEN:
    exit("Error: no bot token provided.\nYou are to set environment variable BOT_TOKEN.\nExititng")

try:
    local_server = TelegramAPIServer.from_base("http://localhost:8081")
    bot = Bot(token=BOT_TOKEN, server=local_server)
except ValidationError:
    print()
    exit("Failed to create bot instance: Invalid Bot Token passed!\nExiting...")
    
dp = Dispatcher(bot)


async def check_mime_type(mime_type):
    return "video" in mime_type
    


@dp.message_handler(content_types=[types.ContentType.DOCUMENT, types.ContentType.VIDEO, types.ContentType.ANIMATION])
async def download_test_file(message: types.Message):
    ans = await message.reply("Got a file from you, processing...")
    file_id = None
    obj = None
    if message.document:
        file_id  =  message.document.file_id 
        obj = message.document
    elif message.video:
        file_id = message.video.file_id
        obj = message.video
    elif message.animation:
        file_id = message.animation.file_id
        obj = message.animation
    if not await check_mime_type(obj.mime_type):
        await message.reply("Seems like this is *not* a video file\.\nRefusing to process\!", parse_mode="MarkdownV2")
        return
    
    #needed to make TG servers to send file to my local bot api server
    await bot.get_file(file_id)
    await ans.answer_document(file_id, caption="Saved this file")


@dp.message_handler(commands=["upload"])
async def upload_test_file(message: types.Message):
    await message.answer_chat_action("upload_document")
    file = InputFile("test_file.mp4", filename="Test_file.mp4")
    res = await message.answer_document(file)

@dp.message_handler(commands=["start", "help"])
async def start_handler(message: types.Message):
    await message.answer_chat_action("typing")
    await message.answer(f"Hi, {message.from_user.full_name}!\nI'm Video Processing Bot being under (not) active development!\nTry sneding me file or use /upload command to check if i can share smth with you...")

@dp.message_handler()
async def message_handle(message: types.Message):
    await message.reply("This message does not seem to be a command\.\nUse */help* command\.\.\.", parse_mode="MarkdownV2")

from aiogram.utils.exceptions import NetworkError
from time import sleep
while (True):
    try:
        print("Attempting to connect...")
        executor.start_polling(dp, skip_updates=True)
        break
    except NetworkError:
        print("Network error occured: reconnecting in 10s.")
        sleep(10)
    
