import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class Crawler:
    def __init__(self, pool):
        assert isinstance(pool, (str, list, tuple)), 'url pool should be str, list or tuple'
        self.pool = pool
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
            }

    def crawl(self, soup):
        sublinks = []
        for links in soup.find_all('a'):
            sublinks.append(str(links.get('href')))
        return sublinks

    def fetch(self, url):
        if not url.startswith('http') or not url.startswith('https'):
            url = 'http://' + url
        logger.info(f'start fetch {url}...')
        content = requests.get(url, headers=self.headers)
        return content

    def parse(self, html_doc):
        soup = BeautifulSoup(html_doc, 'lxml')
        return soup

    def download(self):
        pass

    def get_base_url(self, url):
        result = urlparse(url)
        return urlunparse((result.scheme, result.netloc, '', '', '', ''))


class GithubCrawler(Crawler):
    def __init__(self, *args, output_dir='./crawl_result', **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_md_url(self, soup):
        md_url_pool = set()
        sublinks = self.crawl(soup)
        for link in sublinks:
            if not link.startswith('#') and link.endswith('.md'):
                md_url_pool.add(link)
        return md_url_pool
    
    def _get_row_url(self, soup):
        elements = soup.select('div[role=rowheader] a')
        url_pools = set()
        for element in elements:
            sub_url = element.get('href')
            if '.' not in sub_url.split('/')[-1] and element.text.split() != ['.'] * 2:
                url_pools.add(sub_url)
        return url_pools

    def _get_new_pool(self, url):
        base_url = self.get_base_url(url)
        response = self.fetch(url)
        if response.status_code != 200:
            logger.error(f'fail to fetch {url}, respose status code: {response.status_code}')
        soup = self.parse(response.text)
        return set([base_url + i for i in self._get_md_url(soup)]), set([base_url + i for i in self._get_row_url(soup)])

    def _update_pool(self, url, md_url_pool, url_pool):
        base_url = self.get_base_url(url)
        response = self.fetch(url)
        if response.status_code != 200:
            logger.error(f'fail to fetch {url}, respose status code: {response.status_code}')
        soup = self.parse(response.text)
        md_url_pool.update(set([base_url + i for i in self._get_md_url(soup)]))
        url_pool.update(set([base_url + i for i in self._get_row_url(soup)]))
    
    def get_md_content(self, url):
        response = self.fetch(url)
        if response.status_code != 200:
            logger.error(f'fail to fetch {url}, respose status code: {response.status_code}')
        soup = self.parse(response.text)
        element = soup.select_one('article')
        output = []
        if element is None:
            return url, ''
        for line in element.text.split('\n'):
            if line:
                res = element.find(lambda e: e.text == line and e.name.startswith('h'))
                if res:
                    line = res.name + ':' + line
                output.append(line)
        return url, '\n'.join(output)
                
    def start(self, max_deepth=10, workers=10):
        if isinstance(self.pool, str):
            self.pool = [self.pool]
        md_url_pool = set()
        url_pool = set()
        fetched_url = set()
        
        for url in self.pool:
            self._update_pool(url, md_url_pool, url_pool)
            fetched_url.add(url)
            depth = 0
            while len(url_pool) > 0 and depth < max_deepth:
                logger.info(f'current depth {depth} ...')
                result = []
                mp = multiprocessing.Pool(processes=workers)
                for sub_url in url_pool:
                    if sub_url not in fetched_url:
                        result.append(mp.apply_async(self._get_new_pool, (sub_url, )))
                        fetched_url.add(sub_url)
                mp.close()
                mp.join()
                url_pool = set()
                for res in result:
                    mp, up = res.get()
                    url_pool.update(up)
                    md_url_pool.update(mp)
                depth += 1

        result = []
        mp = multiprocessing.Pool(processes=workers)
        for url in md_url_pool:
            result.append(mp.apply_async(self.get_md_content, (url,)))
        mp.close()
        mp.join()
        idx = 0
        idx_dict = {}
        for res in result:
            url, content = res.get()
            idx_dict[idx] = url
            with open(os.path.join(self.output_dir, f'{idx}.txt'), 'w') as f:
                f.write(content)
            idx += 1
        json.dump(idx_dict, open(os.path.join(self.output_dir, 'index.json'), 'w'))


if __name__ == '__main__':
    c = GithubCrawler(pool='https://github.com/intel/neural-compressor')
    c.start()
    # c.get_md_content('https://github.com/intel/neural-compressor/blob/master/README.md')
