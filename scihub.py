# import requests
# from bs4 import BeautifulSoup
from difflib import get_close_matches
import re
from playwright.sync_api import sync_playwright
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import os

# def normalize(text):
#     """Lowercase, strip punctuation, and remove tags like [BOOK][B]"""
#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)  # remove tags like [BOOK][B]
#     text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
#     return text.strip()

# def download_paper_tool(query: str, url: list) -> str:
#     """
#     Tool to find and return a specific paper based on title and/or author.
#     """
#     if not url:
#         return "No URLs available to search from. Please run a prior research query first."

#     query = normalize(query)
#     matches = []

#     for entry in url:
#         title = normalize(entry['Title'])
#         authors = normalize(entry['Authors'])
#         # print('title',title)
#         # print('author',authors)

#         # Exact title match or partial keyword match
#         if query in title or query in authors:
#             matches.append(entry)
#             continue

#         # Split query into words and look for partial hits
#         if any(qword in title or qword in authors for qword in query.split()):
#             matches.append(entry)

#     # If still no matches, try fuzzy matching on title
#     if not matches:
#         titles = [normalize(entry['Title']) for entry in url]
#         close = get_close_matches(query, titles, n=1, cutoff=0.6)
#         if close:
#             for entry in url:
#                 if normalize(entry['Title']) == close[0]:
#                     matches.append(entry)
#                     break
#     # print('match',matches)
#     if matches:
#         paper = matches[0]
#         print(f"Title: {paper['Title']}\nAuthors: {paper['Authors']}\nURL: {paper['URL']}")
#         return f"Title: {paper['Title']}\nAuthors: {paper['Authors']}\nURL: {paper['URL']}"
#     else:
#         return f"No matching paper found for: {query}"
def _scihub_pdf(doi_or_url):
    print('in scihub #############################')
    scihub_url = 'https://sci-hub.se/'

    # Ensure "pdf" folder exists
    os.makedirs("pdf", exist_ok=True)
    payload = {}
    # # Step 1: Fetch the HTML page from Sci-Hub
    # response = requests.get(f"{scihub_url}/{doi_or_url}")
    # response.raise_for_status()
    headers = {
  'content-security-policy': 'upgrade-insecure-requests',
  'Cookie': '__ddg10_=1748585839; __ddg1_=eCLvGkXIMD11pTqs7WXa; __ddg5_=2b54pkYldVN0HgCX; __ddg8_=KewqA7Ra3chVpaJW; __ddg9_=182.65.3.72; __ddgid_=hW9VEQqg860mgwUE; __ddgmark_=Sr4p2BRfMdEJCfej; refresh=1748585839.6917; session=8fe189a559034ec227566c943f52b5ff'
}
    print('doi $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',doi_or_url)

    # Step 1: Fetch the HTML page from Sci-Hub
    response = requests.request("POST", f"{scihub_url}/{doi_or_url}", headers=headers, data=payload)
    response.raise_for_status()
    print('response###############################',response)
    # Step 2: Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    # print('souppppppppppp',venv/soup)

    # Step 3: Find the <embed> tag with the PDF
    embed = soup.find('embed', id='pdf')
    print('embed 222222222222222222222222222222',embed)
    print('embed 222222222222222222222222222222attrrrrrrrrrrrrr',embed.attrs)
    filename = "pdf/"
    print('scihub 33333333333333333333333 three3333333333333333333333333333')
    if embed and 'src' in embed.attrs:
        pdf_url = embed['src']
    # Fix relative URL
        if pdf_url.startswith('/'):
            pdf_url = scihub_url + pdf_url

            # Step 4: Download the PDF
            pdf_response = requests.get(pdf_url, stream=True)
            pdf_response.raise_for_status()
            preview_bytes = next(pdf_response.iter_content(chunk_size=1024))
            print("First few bytes of the PDF content:", preview_bytes[100:300])
            content_length = int(pdf_response.headers.get('Content-Length', 0))
            print(f"Expected size: {content_length} bytes")
            # Clean filename: remove illegal characters
            safe_filename = re.sub(r'[^\w\-_.]', '_', 'paper12') + ".pdf"
            filepath = os.path.join("pdf", safe_filename)

            with open(filepath, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"PDF downloaded as: {filepath}")
    else:
        print("PDF link not found on Sci-Hub page.")

def download_pdf_with_playwright(doi_or_url):
    scihub_url = "https://sci-hub.se/"
    save_dir = "pdf"
    os.makedirs(save_dir, exist_ok=True)

    with sync_playwright() as p:
        # HEADLESS = False -> You will see the browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print(f"Opening {scihub_url}{doi_or_url} ...")
        page.goto(f"{scihub_url}{doi_or_url}", timeout=60000)

        # Wait for the PDF to load
        page.wait_for_selector("embed#pdf, iframe", timeout=15000)

        # Check for PDF element
        try:
            pdf_url = page.locator("embed#pdf").get_attribute("src")
        except:
            pdf_url = page.locator("iframe").get_attribute("src")

        if not pdf_url:
            raise Exception("PDF embed not found.")

        if pdf_url.startswith("/"):
            pdf_url = scihub_url + pdf_url

        print("PDF link:", pdf_url)
        print("🔍 You can visually verify the PDF in the opened browser.")

        # Pause so you can see the page
        input("Press Enter to continue and download the PDF...")

        page.evaluate(f"""
            const link = document.createElement('a');
            link.href = "{pdf_url}";
            link.download = "";
            document.body.appendChild(link);
            link.click();
        """)

        with page.expect_download() as download_info:
            print("Waiting for download to start...")
        download = download_info.value

        safe_filename = re.sub(r'[^\w\-_.]', '_', doi_or_url[-20:]) + ".pdf"
        download_path = os.path.join(save_dir, safe_filename)
        download.save_as(download_path)

        browser.close()
        return os.path.join(save_dir, safe_filename)
    
def _scihub_pdf_playwright(doi_or_url):
        print("in scihub (playwright)")

        scihub_url = 'https://sci-hub.se/'

        # doi_or_url, title = self._sentence_similarity(query=query, url_list=self.url)
        target_url = f"{scihub_url}{doi_or_url}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, slow_mo=100)
            context = browser.new_context()
            page = context.new_page()

            try:
                print(f"Opening page: {target_url}")
                page.goto(target_url, timeout=90000)

                # Optional: Save page HTML for debugging
                html = page.content()
                with open("scihub_debug.html", "w", encoding="utf-8") as f:
                    f.write(html)

                # Wait for the embed tag that contains the PDF
                page.wait_for_selector("embed#pdf", timeout=60000, state="attached")
                embed_element = page.query_selector("embed#pdf")

                if not embed_element:
                    print("PDF embed not found.")
                    browser.close()
                    return None

                pdf_url = embed_element.get_attribute('src')

                # Fix relative URLs
                if pdf_url.startswith('/'):
                    pdf_url = scihub_url.rstrip('/') + pdf_url

                print(f"PDF URL found: {pdf_url}")

                # Clean the title to make a safe filename
                safe_filename = re.sub(r'[^\w\-_.]', '_', 'paper-1')
                if not safe_filename.endswith('.pdf'):
                    safe_filename += '.pdf'

                download_path = os.path.join("pdf", safe_filename)
                os.makedirs(os.path.dirname(download_path), exist_ok=True)

                # Download the PDF using requests
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Referer": scihub_url,
                }
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()

                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"PDF downloaded as: {download_path}")
                browser.close()
                return download_path

            except Exception as e:
                print(f"Error: {e}")
                browser.close()
                return None
    
# _scihub_pdf('https://link.springer.com/chapter/10.1007/978-3-319-18305-3_1')
download_pdf_with_playwright('https://www.nature.com/articles/nature14539')
            # 'https://www.science.org/doi/abs/10.1126/science.aaa8415'
# def fun(query,url):

#     for i in range(len(url)):
#         print('i',url[i])
        
#         for j in range(len(url[i])):
#             title = normalize(url[i]['Title'])
#             authors = normalize(url[i]['Authors'])
#             # print('jjjjjjjjjjjj',url[i][j])
#             if title in query or authors in query:
#                 print('______________________________')
#                 print(url[i]['Title'])
#                 print(url[i]['URL'])
#                 print('______________________________')


#             print(title)
#             print('o')
#             pass

url = [{'Title': '[PDF][PDF] Machine learning algorithms-a review', 'Authors': 'B Mahesh\xa0- International Journal of Science and Research\xa0…, 2020 - researchgate.net', 'Abstract': '… Here‟sa quick look at some of the commonly used algorithms in machine learning (ML) \nSupervised Learning Supervised learning is the machine learning task of learning a function …', 'URL': 'https://www.researchgate.net/profile/Batta-Mahesh/publication/344717762_Machine_Learning_Algorithms_-A_Review/links/5f8b2365299bf1b53e2d243a/Machine-Learning-Algorithms-A-Review.pdf?eid=5082902844932096t'}, {'Title': '[BOOK][B] Machine learning', 'Authors': 'E Alpaydin - 2021 - books.google.com', 'Abstract': 'MIT presents a concise primer on machine learning—computer programs that learn from \ndata and the basis of applications like voice recognition and driverless cars. No in-depth …', 'URL': 'https://books.google.com/books?hl=en&lr=&id=Eyk5EAAAQBAJ&oi=fnd&pg=PR9&dq=machine+learning&ots=WRwQdh-noS&sig=VjURxGbtZYujTU6XEyLM9Fdazmc'}, {'Title': '[BOOK][B] Machine learning', 'Authors': 'ZH Zhou - 2021 - books.google.com', 'Abstract': '… from data is called learning or training. The … machine learning is to find or approximate \nground-truth. In this book, models are sometimes called learners, which are machine learning …', 'URL': 'https://books.google.com/books?hl=en&lr=&id=ctM-EAAAQBAJ&oi=fnd&pg=PR6&dq=machine+learning&ots=o_MoV8WyYv&sig=-k5Sp-AOUF2UhrFJdblkVYExnSs'}, {'Title': 'Machine learning: Trends, perspectives, and prospects', 'Authors': 'MI Jordan, TM Mitchell\xa0- Science, 2015 - science.org', 'Abstract': '… Machine learning addresses the question of how to build computers that improve … Recent \nprogress in machine learning has been driven both by the development of new learning …', 'URL': 'https://www.science.org/doi/abs/10.1126/science.aaa8415'}, {'Title': 'What is machine learning?', 'Authors': 'I El Naqa, MJ Murphy\xa0- Machine learning in radiation oncology: theory and\xa0…, 2015 - Springer', 'Abstract': '… A machine learning algorithm is a computational process that … This training is the “learning” \npart of machine learning. The … can practice “lifelong” learning as it processes new data and …', 'URL': 'https://link.springer.com/chapter/10.1007/978-3-319-18305-3_1'}]
# x = sentence_similarity('explain "Trends and Perspectives in Machine Learning"',url)
# print(x)
# scihub_url = 'https://sci-hub.se'
# doi_or_url = '10.1097/WNP.0000000000000574'
# scihub(scihub_url,doi_or_url)