import numpy as np
import gym
from agent import *
from low_agent import *
from config import *
from Environment import Env
from state_gae import high_states, mid_states
from mlp_lcz import build_distribution
from Eneryplus_Datasets.base_energy import buildings_height, buildings_window_to_wars

def train(gird_size, agent_high_level, agent_mid_level, agent_low_level, env, n_episode, scale=1):
    rewards_log_high, average_log_high = [], []
    rewards_log_mid, average_log_mid = [], []
    rewards_log_low, average_log_low = [], []
    palnner_energy, palnner_reward = 0, 0
    for i in range(1, n_episode+1):
        planner = []
        state_high, state_mid = env.reset(high_states, mid_states)
        done = False
        state_history_high, action_history_high, done_history_high, reward_history_high = [list(state_high)], [], [], []
        state_history_mid, action_history_mid, done_history_mid, reward_history_mid = [list(state_mid)], [], [], []
        episodic_reward_high, episodic_reward_mid, episodic_reward_low = 0, 0, 0
        label_high = 1
        for gird in range(gird_size-1):
            probs_high, action_high = agent_high_level.act(state_high)

            probs_mid, action_mid = agent_mid_level.act(state_mid)


            _, reward_mid, done_mid, building_counts, mid_lcz = env.step_mid(probs_mid, action_mid, action_high, build_distribution[gird])

            original_distribution_height = {}

            for building_type, count in enumerate(building_counts):

                standard_height = buildings_height[building_type]
                original_distribution_height[building_type] = {}
                for instance in range(count):

                    original_distribution_height[building_type][instance] = standard_height

            state_low = building_counts
            state_history_low, action_history_low, done_history_low, reward_history_low = [], [], [], []
            actions_low_distribution = {}

            action_space = [-2, -1, 0, 1, 2]

            for building_type, count in enumerate(state_low):
                current_height = buildings_height[building_type]
                actions_low_distribution[building_type] = {}

                current_state = list(state_low) + [building_type, current_height]

                probs_low, action_low = agent_low_level.act(current_state, current_height)
                for instance in range(count):
                    if not state_low.any():
                        state_low = [0] * 16

                    chosen_action = np.random.choice(action_space, p=probs_low)

                    actions_low_distribution[building_type][instance] = current_height + chosen_action

                _, reward_low, done_low, _ = env.step_low(building_type, original_distribution_height[building_type], actions_low_distribution, building_counts, buildings_window_to_wars)

                episodic_reward_low += reward_low
                action_history_low.append(action_low)
                done_history_low.append(done_low)
                reward_history_low.append(reward_low * scale)
                current_state = list(state_low) + [building_type+1, current_height+1]
                state_history_low.append(list(current_state))

            state_values_low, log_probs_low, rewards_low, discounted_rewards_low, dones_low = \
                agent_low_level.process_data(state_history_low, action_history_low, reward_history_low, done_history_low, 64)
            agent_low_level.learn(state_values_low, log_probs_low, rewards_low, discounted_rewards_low, dones_low)


            _, reward_high, done_high, _ = env.step_high(action_high, label_high, mid_lcz)
            planner.append(list(building_counts))

            next_state_high, next_state_mid = high_states[gird+1], mid_states[gird+1]

            episodic_reward_high += reward_high
            action_history_high.append(action_high)
            done_history_high.append(done)
            reward_history_high.append(reward_high * scale)
            state_high = next_state_high
            state_history_high.append(list(state_high))

            episodic_reward_mid += reward_mid
            action_history_mid.append(action_mid)
            done_history_mid.append(done)
            reward_history_mid.append(reward_mid * scale)
            state_mid = next_state_mid
            state_history_mid.append(list(state_mid))


        palnner_reward_before = episodic_reward_high + episodic_reward_mid
        if palnner_reward_before > palnner_reward:
            palnner_reward = palnner_reward_before
            # print(palnner_reward)
            print([sum(column) for column in zip(*planner)])
            print(sum([sum(column) for column in zip(*planner)]))

        state_values_high, log_probs_high, rewards_high, discounted_rewards_high, dones_high = \
            agent_high_level.process_data(state_history_high, action_history_high, reward_history_high, done_history_high, 64)
        agent_high_level.learn(state_values_high, log_probs_high, rewards_high, discounted_rewards_high, dones_high)

        state_values_mid, log_probs_mid, rewards_mid, discounted_rewards_mid, dones_mid = \
            agent_mid_level.process_data(state_history_mid, action_history_mid, reward_history_mid, done_history_mid, 64)
        agent_mid_level.learn(state_values_mid, log_probs_mid, rewards_mid, discounted_rewards_mid, dones_mid)



        rewards_log_high.append(episodic_reward_high)
        average_log_high.append(np.mean(rewards_log_high[-100:]))

        rewards_log_mid.append(episodic_reward_mid)
        average_log_mid.append(np.mean(rewards_log_mid[-100:]))

        rewards_log_low.append(episodic_reward_low)
        average_log_low.append(np.mean(rewards_log_low[-100:]))

        # print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward_high, average_log_high[-1]), end='')
        # if i % 100 == 0:
        #     print()
            
    return rewards_high, average_log_high

if __name__ == '__main__':
    gird_size = 5*5
    env = Env()
    agent_high_level = Agent(env.observation_space_high_level.shape[0], env.action_space_high.n, LEARNING_RATE,
                             GAMMA, DEVICE, SHARE, MODE, CRITIC, NORMALIZE)
    agent_mid_level = Agent(env.observation_space_mid_level.shape[0], env.action_space_mid.n, LEARNING_RATE,
                            GAMMA, DEVICE, SHARE, MODE, CRITIC, NORMALIZE)
    agent_low_level = low_Agent(env.observation_space_low_level.shape[0], env.action_space_low.n, LEARNING_RATE,
                            GAMMA, DEVICE, SHARE, MODE, CRITIC, NORMALIZE)
    rewards_log, _ = train(gird_size, agent_high_level,agent_mid_level, agent_low_level, env, RAM_NUM_EPISODE, SCALE)
    # np.save('{}_rewards.npy'.format(RAM_ENV_NAME), rewards_log)