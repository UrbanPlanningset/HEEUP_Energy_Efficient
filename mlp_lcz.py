import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict


pois_df = pd.read_json('Miami.json', orient='records', lines=True)
grid_5x5 = pd.read_csv('grid_5x5.csv')
with open('grid_assignment_5x5.json', 'r') as file:
    grid_assignment_5x5 = json.load(file)

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

build_distribution = prepare_features(grid_assignment_5x5, pois_df)


labels = grid_5x5['LCZ Category'].values

X_train, X_test, y_train, y_test = train_test_split(build_distribution, labels, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=300)
mlp.fit(X_train_scaled, y_train)

