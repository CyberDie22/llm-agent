import httpx
from bs4 import BeautifulSoup
import urllib.parse

from webpage import get_page_source

http_client = httpx.Client(
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'  # Pretend to be Google Chrome on Windows 10
    },
    follow_redirects=True
)

def image_search(query: str) -> dict:
    print("Searching for images:", query)

    search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote(query)}"
    # search_page = http_client.get(search_url).content
    search_page = get_page_source(search_url)
    with open("images.html", "w") as f:
        f.write(search_page)

    soup = BeautifulSoup(search_page, 'html.parser')

    results = {
        'image_results': [],
    }

    # Extract image search results
    for result in soup.find_all('div', class_='ivg-i'):
        if result.find('img') is None:
            continue

        img = result.find('img')
        img_url = img['src']
        if not img_url.startswith('data:image/jpeg'):
            continue

        # TODO: Find a way to get the full-size image URL (in `a` tag: /imgres?imgurl=)

        results['image_results'].append({
            'url': img_url
        })

    return results

def websearch(query: str) -> dict:
    print("Searching the web for:", query)

    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    search_page = http_client.get(search_url)
    with open("search.html", "wb") as f:
        f.write(search_page.content)

    soup = BeautifulSoup(search_page.content, 'html.parser')

    results = {
        'standard_results': [],
    }

    # Extract standard search results
    for result in soup.find_all('div', class_='g'):
        if result.find('a') is None:
            continue

        a_results = result.find_all('a')
        for a in a_results:
            if a.find('h3') is not None:
                title = a.find('h3').text
                url = a['href']

                snippet_div = result.find('div', {'data-sncf': '1,2'})
                if not snippet_div:
                    snippet_div = result.find('div', {'data-sncf': '1'})
                snippet = snippet_div.text if snippet_div else ''

                results['standard_results'].append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
                break

    return results