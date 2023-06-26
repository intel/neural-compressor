import os
import re
import json
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import multiprocessing
import urllib3

urllib3.disable_warnings()

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
        self.fetched_pool = set()

    def get_sublinks(self, soup):
        sublinks = []
        for links in soup.find_all('a'):
            sublinks.append(str(links.get('href')))
        return sublinks
    
    def get_hyperlink(self, soup, base_url):
        sublinks = []
        for links in soup.find_all('a'):
            link = str(links.get('href'))
            if link.startswith('#') or link is None or link == 'None':
                continue
            suffix = link.split('/')[-1]
            if '.' in suffix and suffix.split('.')[-1] not in ['html', 'htmld']:
                continue
            link_parse = urlparse(link)
            base_url_parse = urlparse(base_url)
            if link_parse.path == '':
                continue
            if link_parse.netloc != '':
                # keep crawler works in the same domain
                if link_parse.netloc != base_url_parse.netloc:
                    continue
                sublinks.append(link)
            else:
                sublinks.append(urlunparse((base_url_parse.scheme,
                                           base_url_parse.netloc,
                                           link_parse.path,
                                           link_parse.params,
                                           link_parse.query,
                                           link_parse.fragment)))
        return sublinks

    def fetch(self, url, max_times=5):
        while max_times:
            if not url.startswith('http') or not url.startswith('https'):
                url = 'http://' + url
            logger.info(f'start fetch {url}...')
            try:
                response = requests.get(url, headers=self.headers, verify=False)
            except Exception as e:
                logger.error(f'fail to fetch {url}, cased by {e}')
            if response.status_code != 200:
                logger.error(f'fail to fetch {url}, respose status code: {response.status_code}')
            else:
                return response
            max_times -= 1
        return None

    def process_work(self, sub_url, work):
        response = self.fetch(sub_url)
        if response is None:
            return []
        self.fetched_pool.add(sub_url)
        soup = self.parse(response.text)
        base_url = self.get_base_url(sub_url)
        sublinks = self.get_hyperlink(soup, base_url)
        if work:
            work(sub_url, soup)
        return sublinks

    def crawl(self, pool, work=None, max_depth=10, workers=10):
        url_pool = set()
        for url in pool:
            base_url = self.get_base_url(url)
            response = self.fetch(url)
            soup = self.parse(response.text)
            sublinks = self.get_hyperlink(soup, base_url)
            self.fetched_pool.add(url)
            url_pool.update(sublinks)
            depth = 0
            while len(url_pool) > 0 and depth < max_depth:
                logger.info(f'current depth {depth} ...')
                mp = multiprocessing.Pool(processes=workers)
                results = []
                for sub_url in url_pool:
                    if sub_url not in self.fetched_pool:
                        results.append(mp.apply_async(self.process_work, (sub_url, work)))
                mp.close()
                mp.join()
                url_pool = set()
                for result in results:
                    sublinks = result.get()
                    url_pool.update(sublinks)
                depth += 1
    
    def parse(self, html_doc):
        soup = BeautifulSoup(html_doc, 'lxml')
        return soup

    def download(self, url, file_name):
        logger.info(f'download {url} into {file_name}...')
        try:
            r = requests.get(url, stream=True, headers=self.headers, verify=False)
            f = open(file_name, "wb")
            for chunk in r.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
        except Exception as e:
            logger.error(f'fail to download {url}, cased by {e}')

    def get_base_url(self, url):
        result = urlparse(url)
        return urlunparse((result.scheme, result.netloc, '', '', '', ''))
    
    def clean_text(self, text):
        text = text.strip().replace('\r', '\n')
        text = re.sub(' +', ' ', text)
        text = re.sub('\n+', '\n', text)
        text = text.split('\n')
        return '\n'.join([i for i in text if i and i != ' '])


class GithubCrawler(Crawler):
    def __init__(self, *args, output_dir='./crawl_result', **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_md_url(self, soup):
        md_url_pool = set()
        sublinks = self.get_sublinks(soup)
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
        soup = self.parse(response.text)
        return set([base_url + i for i in self._get_md_url(soup)]), set([base_url + i for i in self._get_row_url(soup)])

    def _update_pool(self, url, md_url_pool, url_pool):
        base_url = self.get_base_url(url)
        response = self.fetch(url)
        soup = self.parse(response.text)
        md_url_pool.update(set([base_url + i for i in self._get_md_url(soup)]))
        url_pool.update(set([base_url + i for i in self._get_row_url(soup)]))
    
    def get_md_content(self, url):
        response = self.fetch(url)
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
                
    def start(self, max_depth=10, workers=10):
        if isinstance(self.pool, str):
            self.pool = [self.pool]
        md_url_pool = set()
        url_pool = set()
        fetched_url = set()
        
        for url in self.pool:
            self._update_pool(url, md_url_pool, url_pool)
            fetched_url.add(url)
            depth = 0
            while len(url_pool) > 0 and depth < max_depth:
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


class CircuitCrawler(Crawler):
    def __init__(self, *args, output_dir='./crawl_result', **kwargs):
        super().__init__(*args, **kwargs)
        self.headers.update({
            "Cache-Control": "max-age=0",
            "Cookie": "access_token=561hoQKWR9dmQmC0SqXHS3uETgpDkxVVdAXPXvEVHJY; _ga=GA1.2.2073446506.1658713529; isManager=N; BadgeType=BB; IDSID=hengguo; CNCampusCode=SHZ; isMenuVisible=1; ajs_user_id=%2240696693%22; ajs_group_id=null; ajs_anonymous_id=%22cf578c2c-07ec-4ac3-b1eb-a1131cf24ff4%22; s_fid=26C8F0AB24F5B476-298199E6962D1F6F; ELQSTATUS=OK; ELOQUA=GUID=906697F39E824DBF885ED50A398B150B; _cs_c=0; _abck=B51103F18F0989A871324DCB7C6A6228~-1~YAAQdqwsF5kmlAGDAQAA038pCwjsSQiWJnovwNl5wtPLCZzkLadrEmXQKx9j++9Eua1pwXeSsMPz2GOl772NQF+sSQDc+ML6qClNJ1jJTNoZ2NCGE7+90w6B3zWTyn8mLJh+L/Upuj4GCh74hk6lpBHYExXMokSFYbaAKLGw2/4vfCWsZ+XXfUEnymWmb+27volGnYjUDbPFD7Pv8JkmC3BSdmImqbgKCb2w3zkmM9Q3OPQ4J82PLuAzHpMi4uBBficY2apnttO099jggehPhvOqSyQFNopkt6lQx65Cb6Ww7SSGUJE2+bsKXZzoTpQQENAlRxoyNNJIEmpFoTHDXbofGNgRhpF0dWbtUbwUFC2FZ8mr30NdoHcpbg==~-1~-1~-1; AMCV_AD2A1C8B53308E600A490D4D%40AdobeOrg=1585540135%7CMCIDTS%7C19252%7CMCMID%7C75310779783706370332434931749805207698%7CMCAAMLH-1663919541%7C3%7CMCAAMB-1663919541%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1663321941s%7CNONE%7CvVersion%7C4.4.0%7CMCCIDH%7C2131965003; adcloud={%22_les_v%22:%22y%2Cintel.com%2C1663316542%22}; mbox=PC#e0741b84312444dfabf1a6673620ca2c.35_0#1724471070|session#e23cd74f92354c4bb56c59d32b9fb2b3#1663316604; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Mar+28+2023+15%3A07%3A05+GMT%2B0800+(China+Standard+Time)&version=202209.2.0&isIABGlobal=false&hosts=&consentId=3cfefc52-a9fe-4d46-a57f-125e1a84a4cb&interactionCount=0&landingPath=https%3A%2F%2Fwww.intel.com%2Fcontent%2Fwww%2Fus%2Fen%2Fdeveloper%2Ftopic-technology%2Fartificial-intelligence%2Fdeep-learning-boost.html&groups=C0001%3A1%2CC0003%3A0%2CC0004%3A0%2CC0002%3A0; intelresearchSTG=5; intelresearchUID=9098949724218M1679987234236; _cs_id=81508dff-8c83-a140-f131-314d094ea9f8.1661226277.40.1679989156.1679989156.1589385054.1695390277819; utag_main=v_id:0182c8cd1abd0095db8d333c6c600506f003a06700978$_sn:26$_se:1$_ss:1$_st:1680741698802$wa_ecid:75310779783706370332434931749805207698$wa_erpm_id:12253843$ses_id:1680739898802%3Bexp-session$_pn:1%3Bexp-session; kndctr_AD2A1C8B53308E600A490D4D_AdobeOrg_identity=CiY3NTMxMDc3OTc4MzcwNjM3MDMzMjQzNDkzMTc0OTgwNTIwNzY5OFIPCKPttMasMBgBKgRKUE4z8AHxk5%2Df9TA%3D; BIGipServerlbauto-prdeppubdisplb-443=!gpPHJMNhbha3tSF9e6x3zjaYWx2wLtcWLBJaE9pIWPjkxR6Hb8ityCoOf5GBp0LHkZvQg10E7qHVY2E=; _gid=GA1.2.2097819079.1686707457; p-s-t-16b2bfbd-e4d0-493d-be07-e98700e46be1-avm-chat-initiated=true; login-token=97d9b5d9-9341-41c7-a130-17f60d0a1997%3ae9f927a0-85a8-40f5-9bf9-080121414803_635081d4628408f00ad8a87eb3087ad0%3acrx.default; JSESSIONID=node0ymlgikgtgbqawkszgx22yceo57956.node0; _gat=1",
            "Referer": "https://login.microsoftonline.com/",
            "Host": "circuit.intel.com",
            "Sec-Ch-Ua": 'Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "Windows",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        })
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/text', exist_ok=True)
        os.makedirs(f'{self.output_dir}/pdf', exist_ok=True)
        self.fetched_pool = set()
    
    def _work(self, url, soup):
        # element = soup.find('body')
        useless_domain = ['content/it']
        for i in useless_domain:
            if i in url:
                return
        element = soup.find('main')
        if not element:
            element = soup.find('body')
        text = element.text
        text = self.clean_text(text)
        text = text + f'\nThe detail information could refer to {url}.'

        file_name = soup.select_one('head > title')
        if file_name:
            file_name = file_name.text
        else:
            file_name = url.split('/')[-1].split('.')[0]
        file_name = file_name.replace('/', '|').replace(' ', '_')
        file_path = f'{self.output_dir}/text/{file_name}.txt'
        idx = 0
        while os.path.exists(file_path):
            file_path = f'{self.output_dir}/text/{file_name}_{idx}.txt'
            idx += 1
        with open(f'{self.output_dir}/text/{file_name}.txt', 'w') as f:
            f.write(text)
        sublinks = self.get_sublinks(soup)
        base_url = self.get_base_url(url)
        for link in sublinks:
            if link.startswith('/'):
                link = base_url + link
            if link.endswith('pdf'):
                file_name = link.split('/')[-1]
                file_name = file_name.replace('/', '|').replace(' ', '_')
                self.download(link, f'{self.output_dir}/pdf/{file_name}')

    def start(self, max_depth=10, workers=10):
        if isinstance(self.pool, str):
            self.pool = [self.pool]
        self.crawl(self.pool, self._work, max_depth=max_depth, workers=workers) 
            

if __name__ == '__main__':
    pool_list = {
        # 'healthcare_benefits': "https://circuit.intel.com/content/entrypage/99f2d344-dec3-47b5-8b76-943ce2d0313d.html",
        "ergonomics-homepage": "https://circuit.intel.com/content/cs/home/ergonomics/ergonomics-homepage.html",
        # "time_off": "https://circuit.intel.com/content/entrypage/2ebbdfe8-43ae-422f-991c-60ec1195b035.html",
        "compensation_at_intel": "https://circuit.intel.com/content/hr/pay/general/compensation-home.html",
        "employment_guideline": "https://circuit.intel.com/content/entrypage/433515a7-514e-43ff-a5d3-9109318ebaab.html",
        "working-at-intel": "https://circuit.intel.com/content/corp/working-at-intel/home.html",
        "EmpLetterRequestProcess_PRC": "https://circuit.intel.com/content/hr/data-mgmt/records/EmpLetterRequestProcess_PRC.html"
    }
    # c = CircuitCrawler(pool='https://circuit.intel.com/content/entrypage/99f2d344-dec3-47b5-8b76-943ce2d0313d.html', 
    #                    output_dir='./crawl_result/healthcare_benefits')
    # c.start(max_depth=2)
    # c.get_md_content('https://github.com/intel/neural-compressor/blob/master/README.md')
    for name, url in pool_list.items():
        print(name)
        c = CircuitCrawler(pool=url, output_dir=f'./crawl_result/{name}')
        c.start(max_depth=2)
