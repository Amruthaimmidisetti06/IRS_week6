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
