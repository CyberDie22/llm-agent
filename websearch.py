import httpx
from bs4 import BeautifulSoup
import urllib.parse

import pathlib

def websearch(query):
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
