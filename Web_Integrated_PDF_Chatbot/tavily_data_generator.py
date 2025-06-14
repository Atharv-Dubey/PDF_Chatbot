
from tavily import TavilyClient


def get_answer_tavily(query):

    api_key = "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"

    client = TavilyClient(api_key=api_key)


    try:
        answer = client.qna_search(query=query)
        return answer
    except Exception as e:
        return 0

