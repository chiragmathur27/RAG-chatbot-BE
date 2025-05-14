import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlencode

def extract_article_data(url, api_key):
    proxy_params = {
        'api_key': api_key,
        'url': url,
    }

    try:
        response = requests.get(
            url='https://proxy.scrapeops.io/v1/',
            params=urlencode(proxy_params),
            timeout=120,
        )
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find("h1")
        if not title_tag:
            return None

        title = title_tag.get_text(strip=True)

        content_divs = soup.find_all("div", class_="text__text__1FZLe")
        content_paragraphs = [
            div.get_text(strip=True)
            for div in content_divs
            if div.has_attr("data-testid") and div["data-testid"].startswith("paragraph-")
        ]
        content = " ".join(content_paragraphs)

        tags_ul = soup.find("ul", class_="tags__list__15bgr")
        tags = []
        if tags_ul:
            li_tags = tags_ul.find_all("li")[1:]
            for li in li_tags:
                try:
                    tag_text = li.find("a").find("span").find("span").get_text(strip=True)
                    tags.append(tag_text)
                except AttributeError:
                    continue

        timestamp = soup.find("time")
        date = timestamp["datetime"][:10] if timestamp and timestamp.has_attr("datetime") else ""

        return {
            "url": url,
            "title": title,
            "content": content,
            "tags": ", ".join(tags),
            "date": date,
        }

    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None


def get_all_article_urls(limit_sitemaps=2):
    index_url = "https://www.reuters.com/arc/outboundfeeds/sitemap-index/?outputType=xml"
    index_response = requests.get(index_url)
    index_soup = BeautifulSoup(index_response.content, 'lxml-xml')
    sitemap_urls = [loc.get_text(strip=True) for loc in index_soup.find_all("loc")]
    
    all_article_urls = []

    for sitemap_url in sitemap_urls[:limit_sitemaps]:
        print(f"Parsing sitemap: {sitemap_url}")
        sitemap_response = requests.get(sitemap_url)
        sitemap_soup = BeautifulSoup(sitemap_response.content, 'lxml-xml')
        for url_tag in sitemap_soup.find_all("url"):
            loc_tag = url_tag.find("loc")
            if loc_tag:
                all_article_urls.append(loc_tag.get_text(strip=True))

    return all_article_urls


if __name__ == "__main__":
    API_KEY = 'e97a935c-eb1f-487e-8fc4-f79c8f83d54f'
    all_articles = []
    article_urls = get_all_article_urls(limit_sitemaps=5)
    print(len(article_urls))
    for i, url in enumerate(article_urls):
        print(f"[{i+1}/{len(article_urls)}] Extracting: {url}")
        article_data = extract_article_data(url, API_KEY)
        print(article_data)
        if article_data:
            all_articles.append(article_data)
        time.sleep(1)

    # Save to JSON
    with open("reuters_articles.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extracted {len(all_articles)} articles and saved to reuters_articles.json")
