import asyncio
import time
import random
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pyrogram import Client, filters
from pyrogram.enums import ParseMode
from pyrogram.types import Message
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from PIL import Image, ImageEnhance
import numpy as np
import torch
import torch.nn as nn
import cv2

API_ID = 29400566
API_HASH = "8fd30dc496aea7c14cf675f59b74ec6f"
BOT_TOKEN = "7876808534:AAEuY6LGXtR0Qfl2umFAh4ifXAPz-rNBfaw"

app = Client("pinterest_scraper_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SuperResolution()
model.load_state_dict(torch.load("super_res_model.pth", map_location=torch.device("cpu")))
model.eval()

def get_image_urls(url, num_images=5, scroll_attempts_limit=150):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=0)
            page.wait_for_selector("img", timeout=60000)
        except PlaywrightTimeout:
            browser.close()
            return []
        img_urls = set()
        scroll_attempts = 0
        while len(img_urls) < num_images and scroll_attempts < scroll_attempts_limit:
            img_elements = page.query_selector_all("img")
            for img in img_elements:
                src = (
                    img.get_attribute("src") or
                    img.get_attribute("data-src") or
                    img.get_attribute("srcset")
                )
                if src and src.startswith("http") and not src.endswith(".svg") and (".jpg" in src or ".jpeg" in src):
                    img_urls.add(src)
                if len(img_urls) >= num_images:
                    break
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(3, 5))
            scroll_attempts += 1
        browser.close()
        return list(img_urls)[:num_images]

def enhance_image_from_url(img_url):
    try:
        response = urllib.request.urlopen(img_url)
        image = Image.open(BytesIO(response.read())).convert("RGB")
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
        image = ImageEnhance.Sharpness(image).enhance(2.0)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Brightness(image).enhance(1.2)
        img_np = np.array(image)
        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            output = model(img_tensor).squeeze(0).permute(1, 2, 0) * 255.0
        final_img = Image.fromarray(np.uint8(output.numpy()))
        return final_img
    except Exception:
        return None

@app.on_message(filters.command("query") & filters.private)
async def query_handler(client: Client, message: Message):
    query = message.text.split(" ", 1)[1] if len(message.text.split()) > 1 else None
    if not query:
        await message.reply_text("❗ Please provide a search query.\nExample: `/query neon city`", parse_mode=ParseMode.MARKDOWN)
        return

    await message.reply_text(f"🔍 Searching Pinterest for: `{query}`", parse_mode=ParseMode.MARKDOWN)
    try:
        url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"
        img_urls = await asyncio.to_thread(get_image_urls, url, 3)
        if not img_urls:
            await message.reply_text("No high-quality images found 😔", parse_mode=ParseMode.MARKDOWN)
            return

        for img_url in img_urls:
            img = await asyncio.to_thread(enhance_image_from_url, img_url)
            if img:
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                buffer.seek(0)
                await client.send_photo(chat_id=message.chat.id, photo=buffer)
    except Exception as e:
        await message.reply_text(f"⚠️ Error occurred:\n`{str(e)}`", parse_mode=ParseMode.MARKDOWN)

app.run()
