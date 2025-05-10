#!/usr/bin/env python3
import asyncio, aiosqlite, time, yaml, datetime as dt, hashlib, re
import feedparser, schedule
from plyer import notification
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

CONFIG_FILE = "config.yml"
DB = "seen.sqlite"

# ——— Load models ———
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0,
    torch_dtype="auto",
    max_length=140, min_length=35, truncation=True,
)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,
    torch_dtype="auto",
    load_in_8bit=True,
)
embedder = SentenceTransformer("all-mpnet-base-v2")

# ——— Helpers ———
async def init_db():
    async with aiosqlite.connect(DB) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS articles (hash TEXT PRIMARY KEY)")
        await db.commit()

def h(url): return hashlib.sha1(url.encode()).hexdigest()
def toast(t, m, u): notification.notify(title=t[:64], message=f"{m}\n{u}", timeout=10)

async def seen(url_hash):
    async with aiosqlite.connect(DB) as db:
        r = await db.execute("SELECT 1 FROM articles WHERE hash=?", (url_hash,))
        return await r.fetchone() is not None

async def mark(url_hash):
    async with aiosqlite.connect(DB) as db:
        await db.execute("INSERT OR IGNORE INTO articles VALUES (?)", (url_hash,))
        await db.commit()

def classify(txt, filters):
    res = classifier(txt, candidate_labels=list(filters), multi_label=True)
    picked = [l for l,s in zip(res["labels"],res["scores"]) if s>0.3]
    return " ".join("#"+l for l in picked) or "#misc"

# ——— Main ———
async def poll():
    await init_db()
    recent = []
    cfg = yaml.safe_load(open(CONFIG_FILE))
    for feed in cfg["feeds"]:
        for e in feedparser.parse(feed).entries[:20]:
            url = e.link; url_h = h(url)
            if await seen(url_h): continue
            title = re.sub(r"\s+"," ", e.title).strip()
            # semantic dedupe
            vec = embedder.encode(title, convert_to_tensor=True)
            if recent and util.cos_sim(vec, recent).max()>0.82: continue
            recent = (recent+[vec])[-60:]
            summary = summarizer(e.get("summary","") or title)[0]["summary_text"]
            tag = classify(title, cfg["filters"])
            toast(f"{tag} {title}", summary, url)
            await mark(url_h)

def main():
    schedule.every(yaml.safe_load(open(CONFIG_FILE))["notify_every"]).seconds.do(
        lambda: asyncio.run(poll())
    )
    asyncio.run(poll())
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__=="__main__":
    main()
