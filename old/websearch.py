import httpx
from bs4 import BeautifulSoup
import urllib.parse

import pathlib

http_client = httpx.Client(
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'  # Pretend to be Google Chrome on Windows 10
    },
    follow_redirects=True
)

def websearch(query: str) -> dict:
    print("Searching the web for:", query)

    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    search_page = http_client.get(search_url)
    with open("../search.html", "wb") as f:
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

def websearch_old(query):
    url_query = urllib.parse.quote(query)

    # TODO: expire cache after 30 minutes or so
    cache_file = pathlib.Path(f"websearch-cache/google/{url_query}.html")
    if cache_file.exists():
        page_source = cache_file.read_text()
    else:
        url = f"https://www.google.com/search?q={url_query}"
        response = httpx.get(url)
        page_source = response.text

        cache_file.write_text(response.text)

    soup = BeautifulSoup(page_source, 'html.parser')

    results = []

    for result in soup.find_all('div', class_='xpd'):
        try:
            inner_divs = result.find_next('div').find_next_siblings('div')
            if len(inner_divs) + 1 != 2:
                continue

            link_a = result.find_next('a')

            link = link_a['href']
            parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
            actual_url = parsed_url.get('q', [None])[0]

            title = link_a.find_next('h3').text

            snippet = inner_divs[0].text

            results.append({
                'title': title,
                'url': actual_url,
                'snippet': snippet
            })
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            continue

    return results
