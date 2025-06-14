import requests
from bs4 import BeautifulSoup
from googlesearch import search


def get_google_results(query, num_results=5):
    urls = list(search(query, num_results=num_results))
    return urls


def extract_first_answer(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        for para in paragraphs:
            text = para.get_text().strip()
            if len(text) > 100:
                return text  # Return the first meaningful paragraph
    except:
        return " "
    return " "


def answer_query(query):
    urls = get_google_results(query)
    state = 0
    text = ""
    for url in urls:
        answer = extract_first_answer(url)
        if answer and "Could not" not in answer:
            state = 1
            text = text + answer
    if state == 1:
        return text
    else:
        return 0
