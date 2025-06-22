import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from utils.preprocessor_for_flowscope import all_simple_paths_of_length_k


def plot_graph(G):
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title("Transaction Graph Visualization")
    plt.axis('off')
    plt.show()

def load_and_build_graph(fs1_path, fs2_path, fs3_path):
    fs1 = pd.read_csv(fs1_path, header=None, names=['source', 'target', 'weight'])
    fs2 = pd.read_csv(fs2_path, header=None, names=['source', 'target', 'weight'])
    fs3 = pd.read_csv(fs3_path, header=None, names=['source', 'target', 'weight'])

    G = nx.DiGraph()

    for _, row in fs1.iterrows():
        G.add_edge(f"L1_{row['source']}", f"L2_{row['target']}", weight=row['weight'])

    for _, row in fs2.iterrows():
        G.add_edge(f"L2_{row['source']}", f"L3_{row['target']}", weight=row['weight'])

    for _, row in fs3.iterrows():
        G.add_edge(f"L3_{row['source']}", f"L4_{row['target']}", weight=row['weight'])

    return G

def compute_positions(G):
    layers = {
        1: [n for n in G.nodes if n.startswith('L1_')],
        2: [n for n in G.nodes if n.startswith('L2_')],
        3: [n for n in G.nodes if n.startswith('L3_')],
        4: [n for n in G.nodes if n.startswith('L4_')],
    }

    pos = {}
    for layer_num, nodes in layers.items():
        y = -layer_num
        x_spacing = 1
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i * x_spacing, y)
    return pos

def visualize_graph(G, pos):
    plt.figure(figsize=(16, 8))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', arrowsize=15)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Multipartite Graph Visualization: Layers 1 to 4")
    plt.axis('off')
    plt.show()

def plot_multipartite_graph_size(k):
    df = pd.read_csv(f"./auxiliary/found-patterns-{k}.csv")
    df['From'] = df['From Bank'].astype(str) + '_' + df['From Account'].astype(str)
    df['To'] = df['To Bank'].astype(str) + '_' + df['To Account'].astype(str)

    G = nx.DiGraph()
    for _,row in df.iterrows():
        G.add_edge(row['From'], row['To'], weight=row['Amount Paid'])

    all_paths = []
    for node in G.nodes:
        current_paths = all_simple_paths_of_length_k(G, node, k)
        all_paths.extend(current_paths)


        H = nx.DiGraph()
        for path in all_paths:
            for layer, node in enumerate(path):
                H.add_node(node, layer=layer)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                weight = G[u][v]['weight'] if G.has_edge(u, v) else 1.0
                H.add_edge(u, v, weight=weight)

        pos = {}
        layers = {}
        for node, data in H.nodes(data=True):
            layer = data['layer']
            layers.setdefault(layer, []).append(node)

        for layer, nodes in layers.items():
            for i, node in enumerate(nodes):
                pos[node] = (i, -layer)

        plt.figure(figsize=(16, 6))
        nx.draw(H, pos, with_labels=False, node_size=300, node_color='skyblue', edge_color='gray')

        edge_labels = nx.get_edge_attributes(H, 'weight')
        formatted_labels = {edge: f"{w:,.2f}" for edge, w in edge_labels.items()}
        nx.draw_networkx_edge_labels(H, pos, edge_labels=formatted_labels, font_size=8)

        plt.title(f"{k + 1}-Partite Graph with Transaction Amounts (k={k})")
        plt.axis('off')
        plt.show()


def main():
    fs1_path = '../encoded-output/fs1.csv'
    fs2_path = '../encoded-output/fs2.csv'
    fs3_path = '../encoded-output/fs3.csv'

    G = load_and_build_graph(fs1_path, fs2_path, fs3_path)
    pos = compute_positions(G)
    visualize_graph(G, pos)

if __name__ == "__main__":
    main()
