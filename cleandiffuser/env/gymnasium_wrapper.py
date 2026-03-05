import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
import dill
from typing import Dict
from mani_skill.utils import common

# ------------------ MultiStepWrapper ------------------------

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
        result[key] = result[key].transpose(1, 0, *range(2, len(result[key].shape)))
    return result

def aggregate(data, method='max'):
    # data is list of arrays (num_envs, ...)
    if method == 'max':
        # equivalent to any
        return np.max(data, axis=0)
    elif method == 'min':
        # equivalent to all
        return np.min(data, axis=0)
    elif method == 'mean':
        return np.mean(data, axis=0)
    elif method == 'sum':
        return np.sum(data, axis=0)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self.real_obs_space = self.env.env.env.env.env.env.env.env.observation_space
        self._observation_space = repeated_space(self.real_obs_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = deque(maxlen=n_obs_steps+1)
    
    def reset(self, seed=None, options=None):
        """Resets the environment using kwargs."""
        obs, raw_info = super().reset(seed=seed, options=options)

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        info = {'success': None}
        for key in info.keys():
            info[key] = raw_info[key].cpu().numpy()
        self.info = deque([info], maxlen=self.n_obs_steps+1)

        observation = self._get_obs(self.n_obs_steps)
        information = self._get_info(self.n_obs_steps)

        for key in observation.keys():
            # n_obs_steps, num_env, ... -> num_env, n_obs_steps, ...
            observation[key] = observation[key].transpose(1, 0, *range(2, len(observation[key].shape)))
            # transform into ..., H, W, C -> ..., C, H, W
            if key.startswith('rgb'):
                observation[key] = observation[key].transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.
        
        for key in information.keys():
            information[key] = information[key].transpose(1, 0, *range(2, len(information[key].shape)))

        return observation, information
    
    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        action = action.transpose(1, 0, 2)
        for act in action:
            # if len(self.done) > 0 and self.done[-1]:
            #     # termination
            #     break
            observation, reward, terminated, truncated, raw_info = super().step(act)
            reward = reward.cpu().numpy()
            terminated = terminated.cpu().numpy()
            truncated = truncated.cpu().numpy()
            info = {'success': None}
            for key in info.keys():
                info[key] = raw_info[key].cpu().numpy()

            self.obs.append(observation)
            self.reward.append(reward)
            done = np.array([terminated[j] or truncated[j] for j in range(len(terminated))])
            # if (self.max_episode_steps is not None) \
            #     and (len(self.reward) >= self.max_episode_steps):
            #     # truncation
            #     done = True
            self.done.append(done)
            self.info.append(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, 'sum')
        done = aggregate(self.done, 'max')
        information = self._get_info(self.n_obs_steps)
        
        for key in observation.keys():
            # n_obs_steps, num_env, ... -> num_env, n_obs_steps, ...
            observation[key] = observation[key].transpose(1, 0, *range(2, len(observation[key].shape)))
            # transform into ..., H, W, C -> ..., C, H, W
            if key.startswith('rgb'):
                observation[key] = observation[key].transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.

        for key in information.keys():
            information[key] = information[key].transpose(1, 0, *range(2, len(information[key].shape)))
        
        return observation, reward, done, information

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')
    
    def _get_info(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.info) > 0)
        result = dict()
        for key in self.info[0].keys():
            result[key] = stack_last_n_obs(
                [info[key] for info in self.info],
                n_steps
            )
        return result

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
    """

    def __init__(self, env, rgb=True, depth=False, state=False, oracle=False, joints=True) -> None:
        # self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
        super().__init__(env)
        self.env = env
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_oracle = oracle
        self.include_joints = joints

        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self.env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        ret = dict()

        if self.include_rgb:
            sensor_data = observation.pop("sensor_data")

            del observation["sensor_param"]
            ret['rgb'] = sensor_data['base_camera']['rgb']
            ret['rgb_wrist'] = sensor_data['hand_camera']['rgb']
        
        if self.include_depth:
            raise NotImplementedError("Depth inclusion not implemented yet.")
        
        if self.include_state:
            raise NotImplementedError("State inclusion not implemented yet.")

        if self.include_oracle:
            ret['oracle'] = observation['oracle_info']

        if self.include_joints:
            # Create extra_agent dict with 'extra' and 'agent' keys
            extra_agent = {}
            for key in ['extra', 'agent']:
                if key in observation:
                    extra_agent[key] = observation.pop(key)

            # Flatten the extra_agent dict
            extra_agent_flat = common.flatten_state_dict(extra_agent, use_torch=True, device=self.env.device)
            ret['joints'] = extra_agent_flat

        ret_numpy = {}
        for k, v in ret.items():
            if isinstance(v, np.ndarray):
                ret_numpy[k] = v
            else:
                ret_numpy[k] = v.cpu().numpy()
        
        return ret_numpy