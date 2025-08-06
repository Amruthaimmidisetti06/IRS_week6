import requests
from bs4 import BeautifulSoup

topic = input("Enter the news topic to search for: ")
websites = input("Enter comma-separated websites to limit crawling (e.g., bbc.com,cnn.com): ").split(',')

SERP_API_KEY = '8d6bc2b3eef2e66c277a5a34be29b70d490834e929934539b15ae91c71dd569c'
search_url = 'https://serpapi.com/search.json'

def search_news(topic, websites):
    all_results = []
    for site in websites:
        params = {
            "engine": "google",
            "q": f"{topic} site:{site.strip()}",
            "api_key": SERP_API_KEY
        }
        response = requests.get(search_url, params=params)
        data = response.json()
        if "organic_results" in data:
            for result in data["organic_results"]:
                title = result.get("title")
                link = result.get("link")
                snippet = result.get("snippet", "")
                all_results.append((title, link, snippet))
    return all_results

def display_results(results):
    for idx, (title, link, snippet) in enumerate(results, start=1):
        print(f"\nNews {idx}:")
        print(f"Title   : {title}")
        print(f"URL     : {link}")
        print(f"Summary : {snippet}")

results = search_news(topic, websites)

if results:
    display_results(results)
else:
    print("No results found.")
