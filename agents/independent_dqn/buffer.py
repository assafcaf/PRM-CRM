class PredictorBuffer:
    def __init__(self, nun_envs):
        self.nun_envs = nun_envs
        self.pathes = [{'obs': [],
                        'actions': [],
                        'rewards': [],
                        'original_rewards': [],
                        'human_obs': []}
                        for _ in range(self.nun_envs)] 
    
    def store(self, obs, actions, pred_rewards, real_rewards, human_obs):
        for i in range( self.nun_envs):
            self.pathes[i]['obs'].append(obs[i])
            self.pathes[i]['actions'].append(actions[i].item())
            self.pathes[i]['rewards'].append(pred_rewards[i])
            self.pathes[i]["original_rewards"].append(real_rewards[i])
            self.pathes[i]["human_obs"].append(human_obs[i])
    
    def get(self):
        pathes = self.pathes
        self.pathes = [{'obs': [],
                        'actions': [],
                        'rewards': [],
                        'original_rewards': [],
                        'human_obs': []}
                        for _ in range(self.nun_envs)]
        for path in pathes:
            yield path
    