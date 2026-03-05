from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key == "past_act":
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space
        self.past_action = np.zeros(7)


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            if "past_act" == key:
                obs[key] = self.past_action
            else:
                obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        self.past_action = np.zeros(7)
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        self.past_action = action
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img

class LongSquareWrapper(RobomimicImageWrapper):
    def __init__(self, 
        env: RobomimicImageWrapper,
        ):

        self.env = env
        # self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.action_space =self.env.action_space
        self.observation_space = self.env.observation_space

    def get_observation(self):
        return self.env.get_observation()

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        # state = self.env.get_state()['states'] 

        nut_pos = np.random.uniform([-0.2, -0.2], [0, 0.2], size=(2))

        reset_state = np.array([ 0.        , -0.02921895,  0.17810908,  0.02728627, -2.63967499, \
                -0.01431297,  2.9502351 ,  0.77126893,  0.020833  , -0.020833  , \
                -0.11083517,  0.11445349,  0.89      ,  -0.98421243,  0.        , # <- first trhee for some value \ 
                    0.        ,  0.17699121, 10.        , 10.        , 10.        , \
                    1.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        ])  

        self.rew = 0
        reset_state[10:12] = nut_pos
        self.env.env.reset_to({"states": reset_state})
        nut_pos = self.env.env.env.sim.data.body_xpos[self.env.env.env.obj_body_id['SquareNut']]
        
        peg_pos1 = np.array(self.env.env.env.sim.data.body_xpos[self.env.env.env.peg1_body_id])
        
        peg_pos2 = np.array(self.env.env.env.sim.data.body_xpos[self.env.env.env.peg2_body_id])


        if np.linalg.norm(nut_pos - peg_pos1) < np.linalg.norm(nut_pos - peg_pos2):
            self.target_peg_id = 1
        else:
            self.target_peg_id = 0

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # TODO change reward
        peg_id = 0
        peg_pos = self.env.env.env.sim.data.body_xpos[self.env.env.env.obj_body_id['SquareNut']]
        on_peg = self.env.env.env.on_peg(peg_pos, peg_id)

        on_saddle_drop = np.linalg.norm(obs["robot0_eef_pos"] - np.array([0,0,0.89])) < 0.03
        on_saddle = np.linalg.norm(obs["robot0_eef_pos"] - np.array([0,0,0.89+0.2])) < 0.09

        if self.rew == 0 and on_peg:
            self.rew += 1
        elif self.rew == 1 and on_saddle:
            self.rew += 1
        elif self.rew == 2 and on_peg:
            self.rew += 1
        elif self.rew == 3 and on_saddle_drop:
            self.rew += 1

        reward = 1 if self.rew == 4 else 0

        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
    

def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/cleandiffuser/cleandiffuser/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/cleandiffuser/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
