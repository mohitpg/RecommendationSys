from flask import Flask, request, jsonify
from flask_cors import CORS
import os.path as osp
from pathlib import Path
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T
from dataset import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = osp.join(Path().resolve(), 'models', 'model3.pt')
path = osp.join(Path().resolve(), 'data')
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

model = Model(hidden_channels=32).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

import pandas as pd
df_movies = pd.read_csv('data/raw/ml-latest-small/movies.csv', index_col='movieId')
df_ratings = pd.read_csv('data/raw/ml-latest-small/ratings.csv')

movie_mapping = {i: idx for i, idx in enumerate(df_movies.index)}
num_movies = len(data['movie'].x)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST']) 
def home_page():
    return "<h1>Movie recommendation api</h1>"

@app.route('/rec', methods=['GET', 'POST']) 
def recommend():
    data_rec = request.get_json()
    USERID=data_rec['userid']
    NUM_MOVIES=data_rec['num']

    row = torch.tensor([USERID] * num_movies)
    col = torch.arange(num_movies)
    edge_label_index = torch.stack([row, col], dim=0)
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
    pred = pred.clamp(min=0, max=5)
    idx_max = torch.topk(pred, NUM_MOVIES).indices
    ret=[]
    for i in idx_max:
        movieId = movie_mapping[int(i)]
        ret.append(df_movies.loc[movieId].title)
    return jsonify(ret)

if __name__ == '__main__':
    app.run()  