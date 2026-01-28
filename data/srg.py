import os
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("Please install networkx: pip install networkx") from e


SRG_SPECS = {
    "SRG_16622": ("sr16622.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr16622.g6"),
    "SRG_251256": ("sr251256.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr251256.g6"),  # (25,12,5,6) 15 graphs
    "SRG_261034": ("sr261034.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr261034.g6"),  # (26,10,3,4) 10 graphs :contentReference[oaicite:3]{index=3}
    "SRG_281264": ("sr281264.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr281264.g6"),  # (28,12,6,4) 4 graphs
    "SRG_291467": ("sr291467.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr291467.g6"),  # (29,14,6,7) 41 graphs :contentReference[oaicite:4]{index=4}
    "SRG_351668": ("sr351668.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr351668.g6"),  # (35,16,6,8) 3854 graphs
    "SRG_351899": ("sr351899.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr351899.g6"),  # (35,18,9,9) 227 graphs
    "SRG_361446": ("sr361446.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr361446.g6"),  # (36,14,4,6) 180 graphs
    "SRG_401224": ("sr401224.g6", "https://users.cecs.anu.edu.au/~bdm/data/sr401224.g6"),  # (40,12,2,4) 28 graphs :contentReference[oaicite:5]{index=5}
}

def _parse_graph6_tokens(path: str):
    with open(path, "rt", encoding="ascii", errors="ignore") as f:
        content = f.read()
    tokens = [tok.strip() for tok in content.split() if tok.strip() and not tok.strip().startswith("#")]
    return tokens

def _g6_to_pyg_data(g6: str, y: int):
    G = nx.from_graph6_bytes(g6.encode("ascii"))
    n = G.number_of_nodes()
    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src = torch.tensor([u for u, v in edges] + [v for u, v in edges], dtype=torch.long)
        dst = torch.tensor([v for u, v in edges] + [u for u, v in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

    data = Data(edge_index=edge_index, y=torch.tensor([y], dtype=torch.long), num_nodes=n)
    return data


class SRGGraphDataset(InMemoryDataset):
    def __init__(self, root: str, name: str, transform=None, pre_transform=None, force_reload: bool = False):
        assert name in SRG_SPECS, f"Unknown SRG name: {name}. Choose from {list(SRG_SPECS.keys())}"
        self.name = name
        self.raw_filename, self.url = SRG_SPECS[name]
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        return [f"data_{self.name}.pt"]

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_filename)
        tokens = _parse_graph6_tokens(raw_path)

        data_list = []
        for i, g6 in enumerate(tokens):
            data = _g6_to_pyg_data(g6, y=i)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])