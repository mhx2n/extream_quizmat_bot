import os
import requests
import json
import time

# ================= CONFIG =================
BOT_TOKEN = "8200161005:AAF_bgiFj7UYVtDGddi3yAT9GW7zFQzBr_U"
OWNER_ID = 8389621809

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

TEMP = "temp"
DB = "db.json"

os.makedirs(TEMP, exist_ok=True)

pending_files = {}

# ================= DB =================
def load_db():
    if not os.path.exists(DB):
        return {}
    with open(DB, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB, "w") as f:
        json.dump(data, f)

def set_thumb(file_path):
    data = load_db()
    data["thumb"] = file_path
    save_db(data)

def get_thumb():
    return load_db().get("thumb")

# ================= TELEGRAM HELPERS =================
def send_message(chat_id, text):
    requests.post(f"{BASE_URL}/sendMessage", json={
        "chat_id": chat_id,
        "text": text
    })

def send_document(chat_id, file_path, thumb_path):
    with open(file_path, "rb") as doc, open(thumb_path, "rb") as thumb:
        requests.post(f"{BASE_URL}/sendDocument", data={
            "chat_id": chat_id,
            "caption": "✅ Done"
        }, files={
            "document": doc,
            "thumbnail": thumb
        })

def get_file_path(file_id):
    r = requests.get(f"{BASE_URL}/getFile?file_id={file_id}").json()
    return r["result"]["file_path"]

def download_file(file_id, save_as):
    file_path = get_file_path(file_id)
    file_url = f"{FILE_URL}/{file_path}"
    r = requests.get(file_url)
    with open(save_as, "wb") as f:
        f.write(r.content)

# ================= MAIN LOOP =================
offset = None

print("Bot Running...")

while True:
    try:
        updates = requests.get(f"{BASE_URL}/getUpdates", params={
            "timeout": 30,
            "offset": offset
        }).json()

        for update in updates["result"]:
            offset = update["update_id"] + 1

            if "message" not in update:
                continue

            message = update["message"]
            chat_id = message["chat"]["id"]

            if chat_id != OWNER_ID:
                continue

            # PHOTO → SAVE THUMB
            if "photo" in message:
                file_id = message["photo"][-1]["file_id"]
                thumb_path = f"{TEMP}/thumb.jpg"
                download_file(file_id, thumb_path)
                set_thumb(thumb_path)
                send_message(chat_id, "✅ Thumbnail Saved")

            # DOCUMENT → ASK RENAME
            elif "document" in message:
                file_id = message["document"]["file_id"]
                file_name = message["document"]["file_name"]

                ext = os.path.splitext(file_name)[1]
                save_path = f"{TEMP}/original{ext}"

                download_file(file_id, save_path)

                pending_files[chat_id] = save_path
                send_message(chat_id, "✏️ Send new file name (without extension)")

            # TEXT → RENAME PROCESS
            elif "text" in message:
                text = message["text"]

                if chat_id in pending_files:
                    old_path = pending_files[chat_id]
                    ext = os.path.splitext(old_path)[1]
                    new_path = f"{TEMP}/{text}{ext}"

                    os.rename(old_path, new_path)

                    thumb = get_thumb()
                    if not thumb:
                        send_message(chat_id, "❌ No thumbnail set.")
                        continue

                    send_message(chat_id, "📤 Uploading...")

                    send_document(chat_id, new_path, thumb)

                    os.remove(new_path)
                    del pending_files[chat_id]

    except Exception as e:
        print("ERROR:", e)

    time.sleep(1)
