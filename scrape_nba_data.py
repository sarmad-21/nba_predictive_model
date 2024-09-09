import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import asyncio

SEASONS = list(range(2010, 2025))  # 09-10, 10-11, ..., 22-23, 23-24
DATA_DIR = "nba_data"
SCHEDULE_DIR = os.path.join(DATA_DIR, "monthly_schedule")
SCORES_DIR = os.path.join(DATA_DIR, "scores")

async def get_html(url, selector, sleep=8, repeat=10):
    html = None
    for i in range(1, repeat+1):
        await asyncio.sleep(sleep*1)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
                await browser.close()
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        except Exception as e:
            print(f"Error on {url}: {e}")
            continue
        else:
            break
    return html

async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    if not html:
        print (f"Failed to retrieve main page for {season} season")
        return
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a")
    href = [l['href'] for l in links if 'href' in l.attrs]
    schedule_pages = [f"https://basketball-reference.com{l}" for l in href]

    for url in schedule_pages:
        save_path = os.path.join(SCHEDULE_DIR, url.split("/")[-1])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            continue
        page_html = await get_html(url, "#all_schedule")
        if page_html:
            with open(save_path, "w+") as f:
                f.write(page_html)
        else:
            print(f"Failed to retrieve {url}")

async def scrape_game(schedule_file):
    try:
        with open(schedule_file, 'r', encoding='utf-8') as f:
            html = f.read()
    except UnicodeDecodeError:
        with open(schedule_file, 'r', encoding='latin1') as f:
            html = f.read()

    soup = BeautifulSoup(html, features="html.parser")
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]

    if not box_scores:
        print(f"No box scores found in {schedule_file}")
        return

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue
        page_html = await get_html(url, "#content")
        if page_html:
            with open(save_path, "w+") as f:
                f.write(page_html)

    else:
        print(f"Failed to retrieve {url}")


async def main():
    for season in SEASONS:
        print(f"Scraping {season} season")
        await scrape_season(season)

    schedule_files = os.listdir(SCHEDULE_DIR)
    print(f"Scraped {len(schedule_files)} schedule files")

    for f in schedule_files:
        filepath = os.path.join(SCHEDULE_DIR, f)
        await scrape_game(filepath)

if __name__ == "__main__":
    asyncio.run(main())



