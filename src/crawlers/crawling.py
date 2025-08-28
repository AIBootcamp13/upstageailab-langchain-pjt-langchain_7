import logging
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader, PdfWriter
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pdf_utf16_text(s: str) -> str:
    # UTF-16BE + BOM 경로로 직렬화되도록 BOM 유니코드 문자 선행
    # pypdf는 PDFDocEncoding 실패 시 UTF-16BE+BOM로 기록한다[웹:433].
    return "\ufeff" + (s or "")

def format_pdf_date(dt: datetime) -> str:
    # 정확한 PDF 날짜 포맷: D:YYYYMMDDHHmmSS (분은 %M, 초는 %S)[웹:439].
    return dt.strftime("D:%Y%m%d%H%M%S")

def write_pdf_metadata_pypdf(src_path: Path, dst_path: Path, title, author, subject, keywords, created_dt, custom=None):
    try:
        reader = PdfReader(str(src_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        # 기존 메타데이터 유지
        if reader.metadata is not None:
            writer.add_metadata(reader.metadata)

        time_str = format_pdf_date(created_dt)

        meta = {
            "/Title":     pdf_utf16_text(title),
            "/Author":    pdf_utf16_text(author),
            "/Subject":   pdf_utf16_text(subject),
            "/Keywords":  pdf_utf16_text(keywords),
            "/Creator":   "crawler-script",
            "/Producer":  "crawler-script",
            "/CreationDate": time_str,
            "/ModDate":      time_str,
        }
        if custom:
            for k, v in custom.items():
                meta[f"/{k}"] = pdf_utf16_text(str(v))

        writer.add_metadata(meta)

        with open(dst_path, "wb") as f:
            writer.write(f)
        logger.info(f"Metadata updated for {dst_path}")
    except Exception as e:
        logger.error(f"Error updating metadata for {src_path}: {str(e)}")

def get_tds_from_table_row(tr):
    try:
        tds = tr.find_all("td")
        ticker = tds[0].get_text(strip=True) if len(tds) >= 1 else ""
        title = tds[1].get_text(strip=True) if len(tds) >= 1 else ""
        date = tds[-2].get_text(strip=True) if len(tds) >= 2 else ""
        company = tds[-4].get_text(strip=True) if len(tds) >= 4 else ""
        date_dt = datetime.strptime(date, "%y.%m.%d") if date else datetime.now()
        return ticker, title, company, date_dt
    except ValueError as e:
        logger.error(f"Date parsing error: {str(e)}")
        return "", "", "", datetime.now()
    except Exception as e:
        logger.error(f"Error extracting table data: {str(e)}")
        return "", "", "", datetime.now()

def get_row_from_a_tag(a_tag):
    tr = a_tag.find_parent("tr")
    if tr is None:
        parent = a_tag.parent
        while parent and parent.name != "tr":
            parent = parent.parent
        tr = parent
    return tr

def get_session_with_retry():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    return session

def get_last_page(session):
    try:
        url = "https://finance.naver.com/research/company_list.naver?page=1"
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paging = soup.find("table", class_="paging")
        if paging:
            last_a = paging.find_all("a")[-1]
            if last_a.text == '맨뒤':
                href = last_a['href']
                last_page = int(href.split('page=')[-1])
                return last_page
        return 1  # 기본값
    except Exception as e:
        logger.error(f"Error getting last page: {str(e)}")
        return 50  # 폴백

def crawling(page, session):
    url = f"https://finance.naver.com/research/company_list.naver?page={page}"
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        pdfs = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if ".pdf" in href:
                if href.startswith("/"):
                    href = "https://finance.naver.com" + href
                tablerow = get_row_from_a_tag(a_tag)
                if tablerow:
                    ticker, title, company, date_dt = get_tds_from_table_row(tablerow)
                    pdfs.append({"link": href, 'ticker': ticker, 'title':title, 'author': company, 'date': date_dt})

        pdfs_dir = Path("data")
        pdfs_dir.mkdir(parents=True, exist_ok=True)

        for pdf in tqdm(pdfs, desc=f"Downloading PDFs from page {page}"):
            file_name = pdf['link'].split("/")[-1]
            file_path = pdfs_dir / file_name
            r = session.get(pdf['link'], stream=True)
            if r.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                write_pdf_metadata_pypdf(
                    src_path=file_path,
                    dst_path=file_path,
                    title=pdf['title'],
                    author=pdf['author'],
                    subject=pdf['ticker'],
                    keywords=pdf['ticker'],
                    created_dt=pdf['date'],
                )
                logger.info(f"Downloaded: {file_name}")
            else:
                logger.warning(f"Failed to download {pdf['link']}: Status {r.status_code}")

        logger.info(f"{page}페이지, PDF 크롤링 완료!")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error crawling page {page}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error on page {page}: {str(e)}")

if __name__ == "__main__":
    session = get_session_with_retry()
    # last_page = get_last_page(session)
    last_page = 10
    logger.info(f"Detected last page: {last_page}")

    for i in tqdm(range(1, last_page + 1), desc="Crawling pages"):
        crawling(i, session)
        time.sleep(5)  # 5초 대기