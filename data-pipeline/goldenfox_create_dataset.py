import asyncio
import json
import os
from urllib.parse import quote_plus
import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tranco import Tranco
from warcio.archiveiterator import ArchiveIterator
from dateparser import parse

# =============== CONFIG =================
SERVER = 'http://index.commoncrawl.org/'
INDEX_NAME = 'CC-MAIN-2025-13'
USER_AGENT = 'mozilla-ai-benchmarks/1.0 (mozilla-fx-ai)'
OUTPUT_FILE = 'goldenfox_dataset_express.jsonl'
READABILITY_JS_PATH = 'Readability.js'
MAX_DOMAINS = 500
MAX_PAGES_PER_DOMAIN = 50
BUFFER_SIZE = 10

# =============== FUNCTIONS ===============

# 1. Get top domains from Tranco
def get_top_domains(limit, golden_domains_path='golden_domains/golden_domains.csv'):
    #t = Tranco(cache=True, cache_dir='.tranco')
    #latest_list = t.list()
    #raw_list = latest_list.top(limit)
    df = pd.read_csv(golden_domains_path)
    raw_list = df['Domain'].values.tolist()


    exclude_domains = ['amazonaws',
    'cloudflare', 'akamai', 'gstatic', 'googleapis', 'root-servers','googlevideo','googletagmanager','outlook.com','msftconnecttest.com',
    'googleusercontent', 'fastly', 'cdn', 'cloudfront', 'akamaiedge',
    'adservice', 'akamaihd', 'msedge', 'edgekey', 'azureedge', 'windowsupdate', 'rubiconproject','scorecardresearch',
    'trafficmanager', 'akamaiedge', 'apple-dns', 'akamai.net', 'cdn77', 'bytefcdn','gmail','doubleclick','adblockplus'
    ]

    filtered_list = []
    for domain in raw_list:
        if domain.endswith('.ru'): # sanctions slava ukraini
            continue
        if any(keyword in domain for keyword in exclude_domains):
            continue
        filtered_list.append(domain)

    return filtered_list

def safe_decode(content):
    if isinstance(content, bytes):
        return content.decode('utf-8', errors='ignore')
    elif isinstance(content, str):
        return content  # Already decoded
    else:
        return ""


# 2. Search Common Crawl Index
def search_commoncrawl(domain):
    encoded_domain = quote_plus(f"{domain}")
    index_url = f'{SERVER}{INDEX_NAME}-index?url={encoded_domain}&matchType=domain&output=json'
    response = requests.get(index_url, headers={'user-agent': USER_AGENT})
    if response.status_code == 200:
        records = response.text.strip().split('\n')
        print(f"{domain} has {len(records)} records")
        return [json.loads(record) for record in records]
    else:
        print("Failed to find on common crawl")
        return []


def is_meaningful_url(url):
    url = url.lower()

    # Reject if obviously broken
    if any(x in url for x in ['%3c', '%3e', '%20','%22', '%27', '%2f%2f', 'http:/http', 'http//http','%d0','%bb%']):
        return False

    bad_patterns = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024',
        'apis.google', 'googleapis', 'doubleclick', 'gstatic', 'adservice', 'adsense',
        'login', 'signup', 'register', 'checkout', 'cart', 'privacy', 'terms', 'account',
        'auth', 'track', 'session', 'sitemap', 'faq', 'redirect', 'goto', '/out/', '/link/', '?url=', '?target=', 'apis.google','googleapis','doubleclick','login', 'signup', 'cart', 'checkout', 'privacy', 'sitemap', 'terms', 'account', 'register', 'search', 'faq','/ads.','redirect',
        'ru.wikinews','/api/','/talk:','/user:','?id=','#','ab.wikipedia','robots.txt'
    ]

    good_patterns = [
        'news', 'article', 'story', 'blog', 'product', 'post', 'doc', 'paper', 'research',
        '2025', 'review', 'guide', 'how-to', 'tutorial', 'whitepaper', 'report','/wiki/', 'details','about','docs','journal','discussion','biography','science','history','item','record',
    ]

    if any(bad in url for bad in bad_patterns):
        return False

    if any(good in url for good in good_patterns):
        return True

    if url.count('/') > 3 and url.count('-') >= 2:
        return True

    if url.endswith('.html') or url.endswith('.htm'):
        return True

    return False


def select_good_records(records, max_per_domain=5, min_crawl_date="20250101"):
    good_records = []
    print(f"Starting with {len(records)}")

    for record in records:
        ts = record.get('timestamp', '')
        if ts and ts[:8] >= min_crawl_date and is_meaningful_url(record.get('url', '')):
            good_records.append(record)

    good_records = sorted(good_records, key=lambda r: r['url'].count('/'), reverse=True)
    print(f"Ending with {len(good_records)}")
    return good_records[:max_per_domain]

# 4. Fetch WARC content
def fetch_html_from_cc(record):
    offset, length = int(record['offset']), int(record['length'])
    s3_url = f"https://data.commoncrawl.org/{record['filename']}"
    byte_range = f'bytes={offset}-{offset+length-1}'
    response = requests.get(s3_url, headers={'user-agent': USER_AGENT, 'Range': byte_range}, stream=True)
    if response.status_code == 206:
        print(f"found it!",s3_url)
        stream = ArchiveIterator(response.raw)
        for warc_record in stream:
            if warc_record.rec_type == 'response':
                output = warc_record.content_stream().read()
                return output
    else:
        print("failed")
    return None


# 5. Extract Readability text using Playwright
async def extract_readable_text(html_content):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content.decode('utf-8', errors='ignore'), wait_until='domcontentloaded')

        with open(READABILITY_JS_PATH, 'r') as f:
            readability_js = f.read()
        await page.add_script_tag(content=readability_js)

        article = await page.evaluate("""
            () => {
                let reader = new Readability(document);
                let article = reader.parse();
                return article ? {text: article.textContent, title: article.title, length: article.length} : null;
            }
        """)

        await browser.close()

        if article is None or article['length'] < 200:
            return None
        return article['text'], article['title']

# 6. Extract meta and OpenGraph metadata
def extract_page_metadata(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    meta_title = soup.title.string.strip() if soup.title and soup.title.string else None

    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_desc_tag['content'].strip() if meta_desc_tag and meta_desc_tag.get('content') else None

    og_title_tag = soup.find('meta', property='og:title')
    og_title = og_title_tag['content'].strip() if og_title_tag and og_title_tag.get('content') else None

    og_desc_tag = soup.find('meta', property='og:description')
    og_description = og_desc_tag['content'].strip() if og_desc_tag and og_desc_tag.get('content') else None

    soup = BeautifulSoup(html, "html.parser")

    date_string = soup.find("meta", {"name": "pubdate"}) or soup.find("meta", {"property": "article:published_time"})

    if date_string and date_string.get("content"):
        publish_date = parse(date_string["content"])

    # Remove common junk
    for script_or_style in soup(['script', 'style', 'noscript']):
        script_or_style.decompose()

    # Get visible text
    text = soup.get_text(separator=' ')

    # Clean whitespace
    text = ' '.join(text.split())

    return {
        "meta_title": meta_title,
        "meta_description": meta_description,
        "og_title": og_title,
        "og_description": og_description,
        "page_text": text,
        "date_string": date_string,
        "publish_date": publish_date if publish_date else None
    }

# 7. Categorize page
def categorize_page(url, text):
    url = url.lower()
    text = text.lower()

    if any(x in url for x in ['news', 'article', 'press']) or 'breaking news' in text:
        return 'News'
    elif any(x in url for x in ['blog', 'opinion', 'post']) or 'subscribe to my blog' in text:
        return 'Blogs'
    elif any(x in url for x in ['product', 'shop', 'item']) or 'add to cart' in text:
        return 'E-commerce'
    elif any(x in url for x in ['forum', 'thread', 'question']) or 'reply' in text:
        return 'Forums'
    elif any(x in url for x in ['paper', 'journal', 'research']) or 'abstract' in text:
        return 'Scientific/Docs'
    elif any(x in url for x in ['landing', 'promo', 'signup']) or 'start your free trial' in text:
        return 'Marketing'
    elif any(x in url for x in ['entertainment', 'movie', 'game']) or 'trailer' in text:
        return 'Entertainment'
    elif any(x in url for x in ['gov', '.gov', 'legislation']) or 'public law' in text:
        return 'Government/Legal'
    elif any(x in url for x in ['edu', 'university', 'course']) or 'enroll' in text:
        return 'Education'
    else:
        return 'Other'

# 8. Curate a single page
async def curate_example(record):
    html_content = fetch_html_from_cc(record)
    if not html_content:
        print("not html")
        return None

    extracted = await extract_readable_text(html_content)
    if not extracted:
        print("not extracted")
        return None

    page_text, readability_title = extracted
    url = record.get('url', 'unknown')
    category = categorize_page(url, page_text)

    page_meta = extract_page_metadata(html_content.decode('utf-8', errors='ignore'))

    return {
        "url": url,
        "extracted_text": page_text,
        "category_major": category,
        "gold_summary": "",  # to annotate later
        "metadata": {
            "crawl_date": record.get('timestamp', 'unknown'),
            "readability_title": readability_title,
            "meta_title": page_meta["meta_title"],
            "meta_description": page_meta["meta_description"],
            "og_title": page_meta["og_title"],
            "og_description": page_meta["og_description"],
            "domain": record.get('url', 'unknown').split('/')[0],
            "page_text_html": page_meta["text"],
        }
    }

# =============== MAIN =================

async def main():
    curated_examples = []
    buffer = []

    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            top_domains = get_top_domains(MAX_DOMAINS)
            print(top_domains)

            for domain in top_domains:
                print(f"Searching domain: {domain}")
                records = search_commoncrawl(domain)
                selected_records = select_good_records(records, MAX_PAGES_PER_DOMAIN)
                print(f"Going to curate {len(selected_records)} records")

                for record in selected_records:
                    print(record['url'])
                    try:
                        example = await curate_example(record)
                        if example:
                            curated_examples.append(example)
                            buffer.append(example)
                            print(f"Curated: {example['url']}")
                            
                            # Flush if buffer full
                            if len(buffer) >= BUFFER_SIZE:
                                for ex in buffer:
                                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                                buffer.clear()

                        else:
                            print("For some reason it failed")
                    except Exception as e:
                        print(f"Error processing {record.get('url')}: {e}")

            # Final flush
            if buffer:
                for ex in buffer:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                buffer.clear()

    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        if buffer:
            for ex in buffer:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            buffer.clear()



    print(f"Done! Collected {len(curated_examples)} examples into {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
