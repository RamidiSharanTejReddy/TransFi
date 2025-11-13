# rag/extractor.py
import asyncio, aiohttp
from bs4 import BeautifulSoup
from yarl import URL
import re
import asyncio
from playwright.async_api import async_playwright

async def fetch_dynamic_content(url: str, timeout: int = 30):
    """Fetch content from dynamic pages using Playwright"""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set timeout and navigate
            page.set_default_timeout(timeout * 1000)
            await page.goto(url, wait_until='networkidle')
            
            # Wait for dynamic content to load
            await page.wait_for_timeout(2000)
            
            # Get page content
            content = await page.content()
            await browser.close()
            
            return url, 200, content
    except Exception as e:
        print(f"Dynamic fetch failed for {url}: {e}")
        return url, 500, ""

async def fetch_with_fallback(session, url: str):
    """Try regular fetch first, fallback to dynamic if needed"""
    # Try regular fetch first
    regular_url, status, content = await fetch(session, url)
    
    if status == 200 and len(content) > 1000:
        return regular_url, status, content
    
    # Fallback to dynamic content fetching
    print(f"🔄 Using dynamic fetch for: {url}")
    return await fetch_dynamic_content(url)

USER_AGENT = 'transfi-rag-bot/0.1'

async def fetch(session: aiohttp.ClientSession, url: str, timeout=30):
    try:
        headers = {
            'User-Agent': USER_AGENT,
            'Accept-Language': 'en-US,en;q=0.9',  # Force English
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        async with session.get(url, timeout=timeout, headers=headers) as resp:
            text = await resp.text()
            return url, resp.status, text
    except Exception as e:
        return url, None, e

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove navigation, footer, and other non-content elements
    for s in soup(['script', 'style', 'noscript', 'iframe', 'nav', 'header', 'footer', 
                   'aside', '.nav', '.navbar', '.footer', '.sidebar', '.menu']):
        s.decompose()
    
    # Focus on main content areas
    main_content = soup.find('main') or soup.find('article') or soup.find('.main-content') or soup.find('#content')
    if main_content:
        soup = main_content
    
    texts = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
        txt = tag.get_text(separator=' ', strip=True)
        if txt and len(txt) > 20:  # Filter out short navigation text
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

def is_english_url(url: str) -> bool:
    """Filter out non-English URLs"""
    lang_patterns = ['/es/', '/id/', '/jp/', '/ph/', '/pt/', '/vi/', '/ja/', '/lt/']
    return not any(pattern in url for pattern in lang_patterns)

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
                full_url = str(url.with_fragment(None))
                if is_english_url(full_url):  # Filter English only
                    links.add(full_url)
        else:
            resolved = str(base.join(url).with_fragment(None))
            if is_english_url(resolved):  # Filter English only
                links.add(resolved)
    if allowed_prefixes:
        links = {u for u in links if any(u.startswith(p) for p in allowed_prefixes)}
    return list(links)
