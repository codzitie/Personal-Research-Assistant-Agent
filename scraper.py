from playwright.sync_api import Playwright
import bs4
from readability import Document
import html2text
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from typing import TypedDict
import asyncio

class Browser:
    def __init__(self, browser):
        print("Starting Chromium in headless mode")
        self.browser = browser

    async def get_page_html(self, url: str):
        """Navigates to the provided URL and returns the HTML"""
        try:
            print("1 . . . . .")
            page = await self.browser.new_page()
            print(f"Browser going to {url}")
            res = await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            if not res or not res.ok:
                print("not ok")
                return ""
            content = await page.content()
            await page.close()
            return content
        except Exception as err:
            print("get except", err)
            return ""


class RetrieverResult(TypedDict):
    url: str
    content: str


class Researcher:
    def __init__(self, get_page_html):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.get_page_html = get_page_html

    async def search_retrieve_content(self, urls: list, query: str = None) -> list[RetrieverResult]:
        retrival_results = []

        for url in urls:
            print("urllll", url['URL'])
            content = await self.get_content(url['URL'])  # AWAIT here
            if content is None:
                continue
            retrival_results.append(content)

        print("revvvv . . . . ", retrival_results)
        return retrival_results

    async def get_content(self, url: str) -> RetrieverResult:
        """Returns the content in a website in a cleaned format"""
        blacklist = ["youtube.com"]
        if any(bll in url for bll in blacklist):
            return None

        page_html = await self.get_page_html(url)  # AWAIT here
        print("pagggghhh", page_html[:300])  # limit print length

        if not page_html:
            return None

        doc = Document(page_html)
        full_html = doc.content()
        content = self.h2t.handle(full_html).replace("\n", " ")
        return {"url": url, "content": content}


async def scrape_it(url_str):
    urls = [{'URL': url_str}]
    async with async_playwright() as playwright:
        browser_instance = await playwright.chromium.launch(headless=False)
        browser = Browser(browser_instance)
        researcher = Researcher(get_page_html=browser.get_page_html)
        results = await researcher.search_retrieve_content(urls)
        return results