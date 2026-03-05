from typing import Dict
import concurrent.futures
from collections import defaultdict

import numpy as np
import torch
import zarr
from tqdm import tqdm

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply
from cleandiffuser.dataset.imagecodecs import register_codecs, Jpeg2k
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, EmptyNormalizer, RotationTransformer, dict_apply, \
    MinMaxNormalizer, ImageNormalizer

register_codecs()

class MikasaEmbDataset(BaseDataset):

    def __init__(
            self,
            dataset_dir: str,
            shape_meta: dict,
            n_obs_steps=None,
            horizon: int = 1,
            pad_before=0,
            pad_after=0,
    ):
        super().__init__()

        self.replay_buffer = _convert_mikasa_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            )
        
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.normalizer = self.get_normalizer()
        

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            if key.endswith("emb"):
                normalizer['obs'][key] = EmptyNormalizer()
            else:
                normalizer['obs'][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer['obs'][key] = ImageNormalizer()
        normalizer['action'] = MinMaxNormalizer(self.replay_buffer['action'][:])

        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # obs
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice], -1, 1
                                        ).astype(np.float32) / 255.
            # T,C,H,W
            del sample[key]
            obs_dict[key] = self.normalizer['obs'][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer['obs'][key].normalize(obs_dict[key])

        # action
        action = sample['action'].astype(np.float32)
        action = self.normalizer['action'].normalize(action)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.tensor),
            'action': torch.tensor(action)
        }
        return torch_data
  
    
def _convert_mikasa_to_replay(store, shape_meta, dataset_path,
                                 n_workers=None, max_inflight_tasks=None):
    """ Convert Mikasa-Robo dataset to ReplayBuffer

    A ReplayBuffer is a `zarr.Group` or Dict[str, dict] that contains the following keys:
    - data: zarr.Group or Dict[str, dict]
        Contains the data. All data should be stored as numpy arrays with the same length.
    - meta: zarr.Group or Dict[str, dict]
        Contains key "episode_ends", which is a numpy array of shape (n_episodes,) that contains the
        end index of each episode in the data.

    Args:
    - store: zarr.Store
        zarr.MemoryStore()
    - shape_meta: dict
        Shape metadata of the dataset. Should contain keys 'obs', 'action'.
        For example:
        shape_meta = {
            "action": {"shape": [10, ]},
            "obs": {
                "rgb": {"shape": [84, 84, 3], "type": "rgb"},
                "robot0_eef_pos":  {"shape": [3, ], "type": "low_dim"},
            }}
    - dataset_path: str
        Path to the Robomimic dataset
    - abs_action: bool
        Whether to use position or velocity control
    - rotation_transformer: RotationTransformer
        Rotation transformer to convert rotation representation
    """

    import multiprocessing
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    # rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
    #     shape = attr['shape']
        type = attr.get('type', 'low_dim')
        assert type == 'low_dim', "Mikasa-Emb dataset only supports low-dim data!"
        lowdim_keys.append(key)
    #     if type == 'rgb':
    #         rgb_keys.append(key)
    #     elif type == 'low_dim':
    #         lowdim_keys.append(key)
    # rgb_keys = ['rgb']
    # lowdim_keys = ['joints']

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    episode_ends = list()
    prev_end = 0
    for i in range(1000):
        ep_data = np.load(f'{dataset_path}/train_data_{i}.npz')
        success = ep_data['success']
        first_success_idx = np.where(success == 1)
        episode_length = first_success_idx[0][0] + 1 if len(first_success_idx[0]) > 0 else 0
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array('episode_ends', episode_ends,
                            dtype=np.int64, compressor=None, overwrite=True)

    # save lowdim data
    for data_key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
        this_data = list()
        for episode_idx in range(1000):
            ep_len = episode_ends[episode_idx] - episode_starts[episode_idx]
            if ep_len == 0:
                continue
            else:
                if data_key == 'action':
                    demo = np.load(f'{dataset_path}/train_data_{episode_idx}.npz')
                    this_data.append(demo[data_key][:ep_len].astype(np.float32))
                else:
                    demo = np.load(f'{dataset_path}/emb_{episode_idx}.npz')
                    this_data.append(demo[data_key][0,:ep_len,:].astype(np.float32))
        this_data = np.concatenate(this_data, axis=0)
        
        if data_key == 'action':
            assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
        else:
            assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][data_key]['shape'])
        
        _ = data_group.array(
            name=data_key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype
        )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer