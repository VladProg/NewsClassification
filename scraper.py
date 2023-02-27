from requests import get
from bs4 import BeautifulSoup
from os import mkdir


def load_page(link):
    print('Loading', repr(link))
    while True:
        try:
            return get(link).text
        except Exception as e:
            print(repr(e))


def read_article(link):
    html = load_page(link)
    html = html[html.index('<div class="newsText">'):
                html.index('<div class="afterNewItemMobileBanner mobileBanner" style="display:none;">')]
    text = BeautifulSoup(html, features='html.parser').get_text().strip()
    text = '\n'.join(' '.join(line.split()) for line in text.split('\n') if not line.startswith('Читайте також: '))
    return text


NEED_LINKS = 1000
INF = 10**9
try:
    mkdir('articles')
except FileExistsError:
    pass
with open('list.txt', 'w') as output:
    for category in ['polytics', 'economy', 'society', 'culture', 'sports', 'technology']:
        articles = set()
        for page in range(1, INF):
            html = load_page(f'https://www.ukrinform.ua/rubric-{category}/block-lastnews?page={page}')
            for part in html.split(f'<a href="/rubric-{category}/')[1:]:
                part = part[: part.index('"')]
                link = f'https://www.ukrinform.ua/rubric-{category}/' + part
                file = 'articles/' + part[: -5] + '.txt'
                articles.add((link, file))
                if len(articles) == NEED_LINKS:
                    break
            if len(articles) == NEED_LINKS:
                break
        for link, file in articles:
            open(file, 'w', encoding='utf-8').write(read_article(link))
            output.write(f'{category:<10} {file}\n')
            output.flush()
