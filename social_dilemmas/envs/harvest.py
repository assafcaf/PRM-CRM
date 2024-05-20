import numpy as np
from numpy.random import rand

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import HARVEST_MAP
from gym.spaces.discrete import Discrete

APPLE_RADIUS = 2

HARVEST_VIEW_SIZE = 7
# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": HARVEST_VIEW_SIZE}  # length of firing range

# SPAWN_PROB = [0, 0.005, 0.02, 0.05] # original values from github
SPAWN_PROB = [0, 0.01, 0.05, 0.01] # values from paper
OUTCAST_POSITION = -99




class HarvestEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        ep_length=1000,
        metric=0,
        same_color=False,
        gray_scale=False
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            same_color=same_color,
            gray_scale=gray_scale
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])
        self.ep_length = ep_length
        self.rewards_record = {}
        self.timeout_record = {}
        self.metric = metric
        self.metrics = {"efficiency": 0,
                        "equality": 0,
                        "sustainability": 0,
                        "peace": 0}
    

    @property
    def action_space(self):
        # return DiscreteWithDType(8, dtype=np.uint8)
        return Discrete(8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        self.metrics = {"efficiency": 0,
                        "equality": 0,
                        "sustainability": 0,
                        "peace": 0}
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def step(self, action):
        observations, rewards, dones, infos= super().step(action)
        for agent_id, _ in self.agents.items():
            infos[agent_id]['r'] = rewards[agent_id]
            infos[agent_id]['fire'] = action[agent_id] == 7
        self.update_social_metrics(rewards)
        return observations, rewards, dones, infos
    
    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(),
            agent.get_orientation(),
            self.all_actions["FIRE"],
            fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

        # Outcast timed-out agents
        for agent_id, agent in self.agents.items():
            if agent.remaining_timeout > 0:
                agent.remaining_timeout -= 1
                # print("Agent %s its on timeout for %d n_steps" % (agent_id, agent.remaining_timeout))
                if not np.any(agent.pos == OUTCAST_POSITION):
                    self.update_map([[agent.pos[0], agent.pos[1], b' ']])
                    agent.pos = np.array([OUTCAST_POSITION, OUTCAST_POSITION])
            # Return agent to environment
            if agent.remaining_timeout == 0 and np.any(agent.pos == OUTCAST_POSITION):
                # print("%s has finished timeout" % agent_id)
                spawn_point = self.spawn_point()
                spawn_rotation = self.spawn_rotation()
                agent.update_agent_pos(spawn_point)
                agent.update_agent_rot(spawn_rotation)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                0 <= x + j < self.world_map.shape[0]
                                and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def update_social_metrics(self, rewards):
        # Save a record of rewards by agent as they are needed for the social metrics computation
        for agent_id, reward in rewards.items():
            if agent_id in self.rewards_record.keys():
                self.rewards_record[agent_id].append(reward)
            else:
                self.rewards_record[agent_id] = [reward]

            is_agent_in_timeout = True if self.agents[agent_id].remaining_timeout > 0 else False
            if agent_id in self.timeout_record.keys():
                self.timeout_record[agent_id].append(is_agent_in_timeout)
            else:
                self.timeout_record[agent_id] = [is_agent_in_timeout]

    def compute_social_metrics(self):
        if len(self.rewards_record) < 1:
            return None

        # Compute sum of rewards
        sum_of_rewards = dict(zip(self.agents.keys(), [0] * self.num_agents))
        for agent_id, rewards in self.rewards_record.items():
            sum_of_rewards[agent_id] = np.sum(rewards)

        agents_sum_rewards = np.sum(list(sum_of_rewards.values()))

        # Compute efficiency/sustainability
        efficiency = agents_sum_rewards / self.num_agents

        # Compute Equality (Gini Coefficient)
        sum_of_diff = 0
        for agent_id_a, rewards_sum_a in sum_of_rewards.items():
            for agent_id_b, rewards_sum_b in sum_of_rewards.items():
                sum_of_diff += np.abs(rewards_sum_a - rewards_sum_b)

        agents_sum_rewards = agents_sum_rewards if agents_sum_rewards != 0 else 1
        equality = 1 - sum_of_diff / (2 * self.num_agents * (agents_sum_rewards))

        # Compute sustainability metric (Average time of at which rewards were collected)
        avg_time = 0
        for agent_id, rewards in self.rewards_record.items():
            pos_reward_time_steps = np.argwhere(np.array(rewards) > 0)
            if pos_reward_time_steps.size != 0:
                avg_time += np.mean(pos_reward_time_steps)

        sustainability = avg_time / (self.num_agents * self.ep_length)

        # Compute peace metric
        timeout_steps = 0
        for agent_id, peace_record in self.timeout_record.items():
            timeout_steps += np.sum(peace_record)
        peace = (self.num_agents * self.ep_length - timeout_steps) / (self.num_agents * self.ep_length)
        metrics = {"efficiency": efficiency,
                   "equality": equality,
                   "sustainability": sustainability,
                   "peace": peace}
        self.metrics = metrics
        self.timeout_record = {}
        self.rewards_record = {}
        
    def get_social_metrics(self):
        return self.metrics
    