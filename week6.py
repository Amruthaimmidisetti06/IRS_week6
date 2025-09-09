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

# ############################################ ## # ############################# ## # ############################

import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Step 1: Parse XML
def parse_xml(xml_text):
    tree = ET.fromstring(xml_text)
    graph = {}
    topics_map = {}

    for page in tree.findall('page'):
        title = page.find('title').text.strip()
        links = [link.text.strip() for link in page.findall('link')]
        topics = page.find('topics').text.strip().split(",") if page.find('topics') is not None else []

        graph[title] = links
        topics_map[title] = [t.strip() for t in topics]

    return graph, topics_map


# Step 2: Build Adjacency Matrix
def build_adj_matrix(graph):
    pages = list(graph.keys())
    idx = {page: i for i, page in enumerate(pages)}
    n = len(pages)
    M = np.zeros((n, n))

    for page, links in graph.items():
        if links:
            for link in links:
                if link in idx:
                    M[idx[link]][idx[page]] = 1 / len(links)
        else:
            M[:, idx[page]] = 1 / n  # dangling node handling

    return M, pages


# Step 3: Compute Topic-Specific PageRank
def topic_specific_pagerank(M, pages, topics_map, topic, d=0.85, tol=1e-6, max_iter=100):
    n = len(pages)
    teleport = np.array([1.0 if topic in topics_map[p] else 0.0 for p in pages])

    if teleport.sum() == 0:
        teleport = np.ones(n)

    teleport = teleport / teleport.sum()  # normalize
    r = np.ones(n) / n  # initial rank

    for _ in range(max_iter):
        r_new = d * M @ r + (1 - d) * teleport
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return dict(zip(pages, r))


# Step 4: Visualize the Web Graph with Topic Highlight
def draw_web_graph(graph, topics_map, topic):
    G = nx.DiGraph()

    for page, links in graph.items():
        for link in links:
            G.add_edge(page, link)

    # Node colors: highlight pages having the topic
    node_colors = [
        "lightgreen" if topic in topics_map.get(page, []) else "skyblue"
        for page in G.nodes()
    ]

    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1500,
        font_size=10,
        arrowsize=15,
        edge_color="gray"
    )
    plt.title(f"Web Graph (Highlighted Topic: {topic})")
    plt.show()


# Step 5: Input and Execute
xml_text = '''<web>
<page>
<title>PageA</title>
<link>PageB</link>
<link>PageC</link>
<topics>science,education</topics>
</page>
<page>
<title>PageB</title>
<link>PageC</link>
<topics>science</topics>
</page>
<page>
<title>PageC</title>
<topics>sports</topics>
</page>
</web>'''

graph, topics_map = parse_xml(xml_text)
M, pages = build_adj_matrix(graph)

# Draw the web graph with topic highlighting
topic = "science"
draw_web_graph(graph, topics_map, topic)

# Compute topic-specific PageRank
ranks = topic_specific_pagerank(M, pages, topics_map, topic)

print("\nTopic-Specific PageRank (Topic: science):")
for page, score in sorted(ranks.items(), key=lambda x: -x[1]):
    print(f"{page}: {score:.4f}")

#################################################################################################################################################################
#################################################################################################################################################################

# week - 8

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load dataset (subset for speed)
categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(newsgroups.data)
print(f"Original TF-IDF shape: {X_tfidf.shape}")  # (docs x terms)

# Step 3: SVD for LSI (Latent Semantic Indexing)
k = 100  # number of latent dimensions
svd = TruncatedSVD(n_components=k)
X_lsi = svd.fit_transform(X_tfidf)
print(f"Reduced LSI shape: {X_lsi.shape}")  # (docs x k topics)

# Step 4: Show similarity between some documents
def show_similar_docs(query_idx, top_n=5):
    similarities = cosine_similarity([X_lsi[query_idx]], X_lsi)[0]
    top_indices = similarities.argsort()[::-1][1:top_n+1]

    print(f"\nQuery Document #{query_idx}:\n{newsgroups.data[query_idx][:300]}...\n")
    print("Top similar documents:")
    for i in top_indices:
        print(f"\nDoc #{i} (Similarity: {similarities[i]:.3f}):\n{newsgroups.data[i][:300]}...")

# Example: Show top 5 similar documents to doc #0
show_similar_docs(query_idx=0, top_n=5)
