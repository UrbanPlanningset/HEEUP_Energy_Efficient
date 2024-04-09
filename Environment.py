import random
from gym import spaces
import numpy as np
from mlp_lcz import mlp, scaler
from gym import spaces
from Eneryplus_Datasets.base_energy import building_energy_usage, building_energy_LCZ, building_energy_height, building_type
from pyenergyplus.api import EnergyPlusAPI


class Env:
    def __init__(self):
        self.action_space_high = spaces.Discrete(16)
        self.action_space_mid = spaces.Discrete(16)
        self.action_space_low = spaces.Discrete(5)

        self.observation_space_high_level = spaces.Box(low=-1, high=1, shape=(128,), dtype=np.float32)

        self.observation_space_mid_level = spaces.Box(low=-1, high=1, shape=(128,), dtype=np.float32)

        self.observation_space_low_level = spaces.Box(low=-1, high=1, shape=(18,), dtype=np.float32)


    def step_high(self, action_high, label_high, mid_lcz, eneygyplus=False):
        if building_energy_LCZ[action_high] == building_energy_LCZ[mid_lcz]:
            reward = 1
        else:
            reward = 1/abs(building_energy_LCZ[action_high] - building_energy_LCZ[mid_lcz])
        done = True
        return None, reward, done, {}


    def step_mid(self, probs_mid, action_mid, aciton_h, n_buildings_types, eneygyplus=False):

        n_buildings = int(sum(n_buildings_types))


        building_types = np.random.choice(a=np.arange(len(probs_mid)), size=n_buildings, p=probs_mid)

        building_counts = np.bincount(building_types, minlength=len(probs_mid))

        total_energy_usage_plan = sum(building_counts[i] * building_energy_usage[i] for i in range(16))

        sample_array = np.array(building_counts).reshape(1, -1)

        sample_scaled = scaler.transform(sample_array)
        label = mlp.predict(scaler.transform(sample_scaled))[0]
        if not eneygyplus:
            reward = 1/(total_energy_usage_plan + abs(building_energy_LCZ[aciton_h] - total_energy_usage_plan))
        else:
            total_energy_usage_plan = sum(building_counts[i] * self.energy_s(building_type[aciton_h]) for i in range(16))
            reward = 1 / (total_energy_usage_plan + abs(building_energy_LCZ[aciton_h] - total_energy_usage_plan))
        done = True
        return None, reward, done, building_counts, label

    def step_low(self, building_type, original_distribution, actions_low_distribution,
                 building_counts, buildings_window_to_wars, eneygyplus=False):


        original_distribution_window_to_wars = {}
        for building_type, count in enumerate(building_counts):
            standard_window_to_wars = buildings_window_to_wars[building_type]
            original_distribution_window_to_wars[building_type] = {}
            for instance in range(count):
                original_distribution_window_to_wars[building_type][instance] = standard_window_to_wars

        after_distribution_window_to_wars = {}

        std_dev = 0.05

        for building_type, count in enumerate(building_counts):
            standard_window_to_wars = buildings_window_to_wars[building_type]
            after_distribution_window_to_wars[building_type] = {}
            for instance in range(count):
                perturbation = np.random.normal(0, std_dev)
                adjusted_window_to_wars = np.clip(standard_window_to_wars + perturbation, 0, 1)
                after_distribution_window_to_wars[building_type][instance] = adjusted_window_to_wars

        diff_dict_height = {}
        for key, value in original_distribution.items():
            if key in actions_low_distribution and isinstance(actions_low_distribution[key], dict):
                diff_dict_height[key] = {nested_key: actions_low_distribution[key][nested_key] - value for nested_key in actions_low_distribution[key]}

        combined_diff_height = {key: sum(nested_values.values()) for key, nested_values in diff_dict_height .items()}
        # print(combined_diff_height)
        if not combined_diff_height:
            reward = 0
        else:
            if not eneygyplus:
                reward = next(iter({key: value * building_energy_height[key] for key, value in combined_diff_height.items()}.values()))
            else:
                reward = next(iter({key: value * self.energy_s(building_type[key]) for key, value in combined_diff_height.items()}.values()))
        done = True
        return None, reward, done, {}


    def energy_s(self, building_type):
        idf_file_name = f'RefBldg+{building_type}+New2004_v1.3_5.0_1A_USA_FL_MIAMI.idf'
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        energy_s = api.runtime.run_energyplus(state, [
            '-d', r'Eneryplus_Datasets',
            '-w', '1A_USA_FL_MIAMI.epw',
            '-r', idf_file_name,
        ])
        return energy_s


    def reset(self, high_states, mid_states):
        state_high = high_states[0]
        state_mid = mid_states[0]
        return state_high, state_mid

    def close(self):
        pass

