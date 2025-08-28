import asyncio
from aiogram import Bot

# Replace with your actual bot token
YOUR_BOT_TOKEN = "8241335689:AAG-Bf-65Jz2r-f2e-1eF-o5b-1rB3s" # Example token

async def delete_webhook():
    bot = Bot(token=YOUR_BOT_TOKEN)
    print("Attempting to delete webhook...")
    if await bot.delete_webhook():
        print("Webhook successfully deleted!")
    else:
        print("No webhook was found or an error occurred.")
    await bot.session.close()

if __name__ == '__main__':
    asyncio.run(delete_webhook())
