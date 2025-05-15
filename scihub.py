# import requests
# from bs4 import BeautifulSoup
from difflib import get_close_matches
import re



def paper_download(query: str, url) -> str:
    """
    Download and process a specific research paper based on query.
    Handles DOI, URL, or natural language queries about papers.
    
    Args:
        query: Search string (DOI, URL, paper title, author name, or combination)
    
    Returns:
        A string containing the paper details and available content
    """
    import re
    from difflib import get_close_matches
    
    # Check if query is empty
    if not query or query.strip() == "":
        return "Error: No paper identifier provided."
    
    # Check if we have any papers in our database
    # if not hasattr(self, 'url') or not url:
    #     return "No paper data available. Please perform a search first."
    
    # Helper function to normalize text for comparison
    def normalize(text):
        """Lowercase, strip punctuation, and remove tags like [BOOK][B]"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)  # remove tags like [BOOK][B]
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        return text.strip()
    
    # Case 1: Input is a DOI
    if re.match(r'^\d+\.\d+\/\S+$', query):
        for paper in url:
            if isinstance(paper, dict):
                # Check if DOI is in the abstract or other fields
                abstract = paper.get('Abstract', '')
                if query in abstract:
                    return _format_paper_result(paper)
        
        return f"No paper with DOI {query} found in the search results."
    
    # Case 2: Input is a URL
    elif query.startswith(('http://', 'https://')):
        for paper in url:
            if isinstance(paper, dict) and paper.get('URL') == query:
                return _format_paper_result(paper)
        
        return f"No paper with URL {query} found in the search results."
    
    # Case 3: Input is a natural language query
    else:
        normalized_query = normalize(query)
        matches = []
        
        # First pass: look for exact or partial matches
        for entry in url:
            if not isinstance(entry, dict):
                continue
                
            title = normalize(entry.get('Title', ''))
            authors = normalize(entry.get('Authors', ''))
            
            # Strong match: query terms found in both title and authors
            if normalized_query in title and normalized_query in authors:
                matches.append((entry, 3))  # Priority 3 (highest)
                continue
                
            # Medium match: query fully contained in title
            if normalized_query in title:
                matches.append((entry, 2))  # Priority 2
                continue
                
            # Weak match: at least one query word in title or authors
            query_words = normalized_query.split()
            if any(qword in title or qword in authors for qword in query_words if len(qword) > 2):
                matches.append((entry, 1))  # Priority 1
        
        # If no matches, try fuzzy matching on title
        if not matches:
            titles = [(normalize(entry.get('Title', '')), entry) for entry in url if isinstance(entry, dict)]
            title_texts = [t[0] for t in titles]
            close_matches = get_close_matches(normalized_query, title_texts, n=3, cutoff=0.6)
            
            if close_matches:
                for close in close_matches:
                    for title, entry in titles:
                        if title == close:
                            matches.append((entry, 0))  # Priority 0 (lowest, fuzzy match)
                            break
        
        # Sort matches by priority (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if matches:
            best_match = matches[0][0]
            return _format_paper_result(best_match)
        else:
            return f"No matching paper found for: {query}"
    
def _format_paper_result( paper):
    """
    Format paper data into a readable result
    
    Args:
        paper: Dictionary containing paper data
    
    Returns:
        Formatted string with paper details
    """
    result = "Found paper:\n"
    result += f"Title: {paper.get('Title', 'Unknown Title')}\n"
    result += f"Authors: {paper.get('Authors', 'Unknown Authors')}\n"
    
    if 'URL' in paper:
        result += f"URL: {paper['URL']}\n"
    
    if 'Abstract' in paper:
        result += f"\nAbstract: {paper['Abstract']}\n"
    
    return result

url = [{'Title': '[PDF][PDF] Machine learning algorithms-a review', 'Authors': 'B Mahesh\xa0- International Journal of Science and Research\xa0…, 2020 - researchgate.net', 'Abstract': '… Here‟sa quick look at some of the commonly used algorithms in machine learning (ML) \nSupervised Learning Supervised learning is the machine learning task of learning a function …', 'URL': 'https://www.researchgate.net/profile/Batta-Mahesh/publication/344717762_Machine_Learning_Algorithms_-A_Review/links/5f8b2365299bf1b53e2d243a/Machine-Learning-Algorithms-A-Review.pdf?eid=5082902844932096t'}, {'Title': '[BOOK][B] Machine learning', 'Authors': 'E Alpaydin - 2021 - books.google.com', 'Abstract': 'MIT presents a concise primer on machine learning—computer programs that learn from \ndata and the basis of applications like voice recognition and driverless cars. No in-depth …', 'URL': 'https://books.google.com/books?hl=en&lr=&id=Eyk5EAAAQBAJ&oi=fnd&pg=PR9&dq=machine+learning&ots=WRwQdh-noS&sig=VjURxGbtZYujTU6XEyLM9Fdazmc'}, {'Title': '[BOOK][B] Machine learning', 'Authors': 'ZH Zhou - 2021 - books.google.com', 'Abstract': '… from data is called learning or training. The … machine learning is to find or approximate \nground-truth. In this book, models are sometimes called learners, which are machine learning …', 'URL': 'https://books.google.com/books?hl=en&lr=&id=ctM-EAAAQBAJ&oi=fnd&pg=PR6&dq=machine+learning&ots=o_MoV8WyYv&sig=-k5Sp-AOUF2UhrFJdblkVYExnSs'}, {'Title': 'Machine learning: Trends, perspectives, and prospects', 'Authors': 'MI Jordan, TM Mitchell\xa0- Science, 2015 - science.org', 'Abstract': '… Machine learning addresses the question of how to build computers that improve … Recent \nprogress in machine learning has been driven both by the development of new learning …', 'URL': 'https://www.science.org/doi/abs/10.1126/science.aaa8415'}, {'Title': 'What is machine learning?', 'Authors': 'I El Naqa, MJ Murphy\xa0- Machine learning in radiation oncology: theory and\xa0…, 2015 - Springer', 'Abstract': '… A machine learning algorithm is a computational process that … This training is the “learning” \npart of machine learning. The … can practice “lifelong” learning as it processes new data and …', 'URL': 'https://link.springer.com/chapter/10.1007/978-3-319-18305-3_1'}]
x = paper_download('explain "Trends and Perspectives in Machine Learning"by MI Jordan',url)
print(x)
# scihub_url = 'https://sci-hub.se'
# doi_or_url = '10.1097/WNP.0000000000000574'
# scihub(scihub_url,doi_or_url)