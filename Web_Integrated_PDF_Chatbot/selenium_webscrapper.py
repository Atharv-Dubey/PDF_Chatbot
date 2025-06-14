from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from urllib.parse import quote_plus

def google_search(query):
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    try:
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        driver.get(search_url)
        time.sleep(50)

        try:
            answer = driver.find_element(By.CLASS_NAME, "Z0LcW")
            return answer.text
        except:
            pass


        try:
            answer = driver.find_element(By.CLASS_NAME, "kno-rdesc")
            return answer.text
        except:
            pass


        snippets = driver.find_elements(By.CLASS_NAME, "VwiC3b")
        for snippet in snippets:
            if len(snippet.text.strip()) > 50:
                return snippet.text.strip()

        return 0

    finally:
        driver.quit()


query = "What is the capital of Finland?"
print(google_search(query))
