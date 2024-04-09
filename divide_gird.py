import pandas as pd
import numpy as np
import json

pois_df = pd.read_json('Miami.json', orient='records', lines=True)

def divide_grid(lat_min, lat_max, lon_min, lon_max, grid_size):
    """
    Divide the area defined by the latitude and longitude bounds into smaller grids.

    Parameters:
    - lat_min, lat_max: The minimum and maximum latitudes.
    - lon_min, lon_max: The minimum and maximum longitudes.
    - grid_size: The size of the grid (e.g., 5x5, 10x10).

    Returns:
    - A DataFrame containing the grid information including the bounds for each grid cell.
    """
    # Calculate the number of divisions for latitude and longitude
    lat_divisions = grid_size
    lon_divisions = grid_size

    # Calculate step sizes for each division
    lat_step = (lat_max - lat_min) / lat_divisions
    lon_step = (lon_max - lon_min) / lon_divisions

    # Generate the grid bounds
    grid_data = []
    for lat_div in range(lat_divisions):
        for lon_div in range(lon_divisions):
            grid_lat_min = lat_min + lat_div * lat_step
            grid_lat_max = grid_lat_min + lat_step
            grid_lon_min = lon_min + lon_div * lon_step
            grid_lon_max = grid_lon_min + lon_step
            grid_data.append([grid_lat_min, grid_lat_max, grid_lon_min, grid_lon_max])

    # Create a DataFrame
    grid_df = pd.DataFrame(grid_data, columns=['Lat Min', 'Lat Max', 'Lon Min', 'Lon Max'])
    return grid_df


# Define the latitude and longitude bounds
lat_min, lat_max = 25.7575, 25.8947
lon_min, lon_max = -80.3581, -80.1868

# Divide the area
grid_5x5 = divide_grid(lat_min, lat_max, lon_min, lon_max, 5)
grid_10x10 = divide_grid(lat_min, lat_max, lon_min, lon_max, 10)
grid_15x15 = divide_grid(lat_min, lat_max, lon_min, lon_max, 15)
grid_20x20 = divide_grid(lat_min, lat_max, lon_min, lon_max, 20)
grid_25x25 = divide_grid(lat_min, lat_max, lon_min, lon_max, 25)
grid_30x30 = divide_grid(lat_min, lat_max, lon_min, lon_max, 30)



def assign_pois_to_grid(pois_df, grid_df):
    """
    Assign each POI to a grid cell based on its latitude and longitude.

    Parameters:
    - pois_df: DataFrame containing the POIs with their latitudes and longitudes.
    - grid_df: DataFrame containing the grid information.

    Returns:
    - A dictionary where keys are grid cell indexes and values are lists of POI indexes belonging to each grid cell.
    """
    # Initialize an empty dictionary to hold the grid assignment for each POI
    grid_assignment = {i: [] for i in range(len(grid_df))}
    # Iterate over each POI to determine its grid assignment
    for poi_index, poi in pois_df.iterrows():
        for grid_index, grid_cell in grid_df.iterrows():
            if (grid_cell['Lat Min'] <= poi['Latitude'] <= grid_cell['Lat Max']) and \
                    (grid_cell['Lon Min'] <= poi['Longitude'] <= grid_cell['Lon Max']):
                grid_assignment[grid_index].append(poi_index)
                break  # Once assigned, no need to check the remaining grid cells
    return grid_assignment

grid_assignment_5x5 = assign_pois_to_grid(pois_df, grid_5x5)
grid_assignment_10x10 = assign_pois_to_grid(pois_df, grid_10x10)
grid_assignment_15x15 = assign_pois_to_grid(pois_df, grid_15x15)
grid_assignment_20x20 = assign_pois_to_grid(pois_df, grid_20x20)
grid_assignment_25x25 = assign_pois_to_grid(pois_df, grid_25x25)
grid_assignment_30x30 = assign_pois_to_grid(pois_df, grid_30x30)

