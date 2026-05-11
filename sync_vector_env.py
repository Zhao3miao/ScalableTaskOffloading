import numpy as np

class MARLSyncVectorEnv:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.max_time = self.envs[0].max_time

    def reset(self):
        obs_list, infos_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos_list.append(info)
        return np.array(obs_list), self._combine_infos(infos_list)

    def step(self, actions_list):
        obs_list, rewards_list, terms_list, truncs_list, infos_list = [], [], [], [], []
        for env, action in zip(self.envs, actions_list):
            obs, reward, term, trunc, info = env.step(action)
            obs_list.append(obs)
            rewards_list.append(reward)
            terms_list.append(term)
            truncs_list.append(trunc)
            infos_list.append(info)

        return (
            np.array(obs_list),
            np.array(rewards_list),
            np.array(terms_list),
            np.array(truncs_list),
            self._combine_infos(infos_list),
        )

    def _combine_infos(self, infos_list):
        combined = {}
        if not infos_list: return combined
        for key in infos_list[0].keys():
            combined[key] = [info.get(key) for info in infos_list]
        return combined

    def close(self):
        for env in self.envs:
            env.close()