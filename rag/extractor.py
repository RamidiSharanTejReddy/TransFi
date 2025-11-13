# rag/extractor.py (same as before)
import asyncio, aiohttp
from bs4 import BeautifulSoup
from yarl import URL
import re

USER_AGENT = 'transfi-rag-bot/0.1'

async def fetch(session: aiohttp.ClientSession, url: str, timeout=30):
    try:
        async with session.get(url, timeout=timeout, headers={'User-Agent': USER_AGENT}) as resp:
            text = await resp.text()
            return url, resp.status, text
    except Exception as e:
        return url, None, e

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for s in soup(['script','style','noscript','iframe']):
        s.decompose()
    texts = []
    for tag in soup.find_all(['h1','h2','h3','h4','p','li']):
        txt = tag.get_text(separator=' ', strip=True)
        if txt:
            texts.append(txt)
    content = '\n\n'.join(texts)
    content = re.sub(r'\s+', ' ', content).strip()
    return content

def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    if soup.title:
        return soup.title.string.strip()
    h1 = soup.find('h1')
    return h1.get_text(strip=True) if h1 else ''

def find_internal_links(html: str, base_url: str, allowed_prefixes=None):
    soup = BeautifulSoup(html, 'html.parser')
    base = URL(base_url)
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('mailto:') or href.startswith('tel:'):
            continue
        url = URL(href)
        if url.is_absolute():
            if url.host == base.host:
                links.add(str(url.with_fragment(None)))
        else:
            resolved = str(base.join(url).with_fragment(None))
            links.add(resolved)
    if allowed_prefixes:
        links = {u for u in links if any(u.startswith(p) for p in allowed_prefixes)}
    return list(links)
