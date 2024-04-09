import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, VGAE
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import VGAE

pois_df = pd.read_json('Miami.json', orient='records', lines=True)
# Defining the latitude and longitude bounds
lat_min, lat_max = 25.7575, 25.8947
lon_min, lon_max = -80.3581, -80.1868
# Defining 16 random types for POIs
poi_types = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15
]

with open('grid_assignment_5x5.json', 'r') as file:
    grid_assignment_5x5 = json.load(file)

# Initialize an empty DataFrame to hold the features for each grid cell
grid_features = pd.DataFrame(0, index=range(len(grid_assignment_5x5)),
                             columns=['Avg Check-in Rate', 'Avg Review Count', 'Avg Star Rating',
                                      'Avg Check-in Rate Again'] + [f'POI Type {i}' for i in range(16)])

# Calculate the features for each grid cell
for grid_index_str, pois_indices in grid_assignment_5x5.items():
    grid_index = int(grid_index_str)
    if 0 <= grid_index < len(grid_features):

        if pois_indices:
            subset = pois_df.iloc[pois_indices]

            for poi_type in poi_types:
                grid_features.loc[grid_index, f'POI Type {poi_type}'] = (subset['Type'] == poi_type).mean()
    else:
        print(f"Warning: Grid index {grid_index} is out of range.")

grid_features.head()
high_features = StandardScaler().fit_transform(grid_features)
high_features = torch.tensor(high_features, dtype=torch.float)
# print(high_features.shape)

def prepare_features(grid_assignment, pois_df, n_types=16):
    features = np.zeros((len(grid_assignment), n_types))
    for grid_index_str, pois_indices in grid_assignment.items():

        grid_index = int(grid_index_str)
        for poi_index_str in pois_indices:

            poi_index = int(poi_index_str)
            poi_type = pois_df.at[poi_index, 'Type']
            if not np.isnan(poi_type):
                features[grid_index, int(poi_type)] += 1
    return features



for i in range(16):
    grid_features[f'Building Type Dist {i}'] = 0
mid_build_distribution = prepare_features(grid_assignment_5x5, pois_df)

for grid_index in range(len(grid_features)):
    for type_index in range(16):

        grid_features.at[grid_index, f'Building Type Dist {type_index}'] = mid_build_distribution[grid_index, type_index]


mid_features = StandardScaler().fit_transform(grid_features)
mid_features = torch.tensor(mid_features, dtype=torch.float)

edge_index = []
for i in range(len(grid_assignment_5x5)):
    for j in range(len(grid_assignment_5x5)):
        if abs(i - j) == 1 or abs(i - j) == 2:  # 假设是5x5网格，简化为线性邻接
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


# 创建图数据对象
data = Data(x=high_features, edge_index=edge_index)




class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 4 * out_channels, heads=8, dropout=0.6)
        # Assuming the output of conv1 is reshaped or concatenated, we adjust the input size for conv2 accordingly
        self.conv2 = GATConv(32 * out_channels, 2 * out_channels, heads=1, concat=True, dropout=0.6)  # Output size doubled for mu and logstd

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Split the output into mu and logstd
        mu = x[:, :x.size(1)//2]
        logstd = x[:, x.size(1)//2:]
        return mu, logstd



encoder = GATEncoder(in_channels=high_features.shape[1], out_channels=128)

model = VGAE(encoder=encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index) + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()

model.eval()
with torch.no_grad():
    high_states = model.encode(data.x, data.edge_index)









# 构建图的边缘索引
# 这需要根据实际的格子邻接关系来确定，这里只是一个概念性的示例
edge_index = []
for i in range(len(grid_assignment_5x5)):
    for j in range(len(grid_assignment_5x5)):
        if abs(i - j) == 1 or abs(i - j) == 2:  # 假设是5x5网格，简化为线性邻接
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
# print(edge_index)

# 创建图数据对象
data = Data(x=mid_features, edge_index=edge_index)
# print(data)

from torch_geometric.nn import GATConv, GCNConv

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 4 * out_channels, heads=8, dropout=0.6)
        # Assuming the output of conv1 is reshaped or concatenated, we adjust the input size for conv2 accordingly
        self.conv2 = GATConv(32 * out_channels, 2 * out_channels, heads=1, concat=True, dropout=0.6)  # Output size doubled for mu and logstd

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Split the output into mu and logstd
        mu = x[:, :x.size(1)//2]
        logstd = x[:, x.size(1)//2:]
        return mu, logstd


# 初始化编码器
encoder = GATEncoder(in_channels=mid_features.shape[1], out_channels=128)
# 创建GAE模型
model = VGAE(encoder=encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index) + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

# 训练循环
for epoch in range(200):
    loss = train()
    # (f'Epoch: {epoch+1}, Loss: {loss:.4f}')

model.eval()
with torch.no_grad():
    # 获取每个节点的嵌入
    mid_states = model.encode(data.x, data.edge_index)
