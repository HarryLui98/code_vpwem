from typing import Dict
import concurrent.futures
from collections import defaultdict
import os

import numpy as np
import torch
import zarr
from tqdm import tqdm

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.imagecodecs import register_codecs, Jpeg2k
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import LongMemSequenceSampler, EmptyNormalizer, \
    MinMaxNormalizer, ImageNormalizer, dict_apply

register_codecs()

class LongMemMikasaDataset(BaseDataset):
    """
    LongMem version of MikasaDataset that supports memory-based sampling.
    Based on LongMemRobomimicImageDataset structure.
    """

    def __init__(
            self,
            dataset_dir: str,
            shape_meta: dict,
            n_obs_steps=None,
            horizon: int = 1,
            pad_before=0,
            pad_after=0,
            dataset_memory_type='random_from_full',
            subsample_ratio=10,
            device='cpu',
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
        
        self.sampler = LongMemSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
            memory_type=dataset_memory_type,
            subsample_ratio=subsample_ratio,
        )
        
        self.subsample_ratio = subsample_ratio

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.device = device

        # Delay normalizer creation to avoid CUDA initialization in forked subprocess
        # Normalizer will be created lazily or set after DataLoader creation
        # Initialize normalizer on CPU first to allow fork-based multiprocessing
        self.normalizer = self.get_normalizer(device=self.device)

    def get_normalizer(self, device=None):
        """
        Create normalizer for the dataset.
        
        Args:
            device: Device to create normalizer on. If None, uses self.device.
                   Use 'cpu' during initialization to allow fork-based multiprocessing.
        """
        if device is None:
            device = self.device
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            if key.endswith("emb"):
                normalizer['obs'][key] = EmptyNormalizer()
            else:
                normalizer['obs'][key] = MinMaxNormalizer(self.replay_buffer[key][:], device=device)
        for key in self.rgb_keys:
            normalizer['obs'][key] = ImageNormalizer(device=device)
        normalizer['action'] = MinMaxNormalizer(self.replay_buffer['action'][:], device=device)

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
            # obs_dict[key] = self.normalizer['obs'][key].normalize(obs_dict[key])

            mem_key = key + "_mem"
            obs_dict[mem_key] = np.moveaxis(sample[mem_key], -1, 1
                                           ).astype(np.float32) / 255.
            del sample[mem_key]
            # obs_dict[mem_key] = self.normalizer['obs'][key].normalize(obs_dict[mem_key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            # obs_dict[key] = self.normalizer['obs'][key].normalize(obs_dict[key])

            mem_key = key + "_mem"
            obs_dict[mem_key] = sample[mem_key].astype(np.float32)
            # obs_dict[mem_key] = self.normalizer['obs'][key].normalize(obs_dict[mem_key])

        obs_dict['ep_step'] = sample['ep_step']
        
        # Store actual_num_frames for dynamic batching
        if 'actual_num_frames' in sample:
            obs_dict['actual_num_frames'] = sample['actual_num_frames']
        else:
            # Fallback: calculate from ep_step length
            obs_dict['actual_num_frames'] = len(sample['ep_step'])

        # action
        action = sample['action'].astype(np.float32)
        # action = self.normalizer['action'].normalize(action)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.tensor),
            'action': torch.tensor(action)
        }
        return torch_data
    
    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        # Return a sampler that sorts by actual_num_frames
        from cleandiffuser.dataset.dataset_utils import SortedByLengthSampler
        return SortedByLengthSampler(self)

    
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
        Path to the Mikasa dataset
    """

    import multiprocessing
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    episode_ends = list()
    prev_end = 0
    max_episodes = 1000
    for i in range(max_episodes):
        train_data_path = f'{dataset_path}/train_data_{i}.npz'
        if not os.path.exists(train_data_path):
            break
        try:
            ep_data = np.load(train_data_path)
            success = ep_data['success']
            first_success_idx = np.where(success == 1)
            episode_length = first_success_idx[0][0] + 1 if len(first_success_idx[0]) > 0 else 0
            episode_length = min(episode_length + 8, 90) if episode_length > 0 else 0
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        except Exception as e:
            break
    
    if len(episode_ends) == 0:
        raise ValueError(f"No episodes found in {dataset_path}")
    
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array('episode_ends', episode_ends,
                            dtype=np.int64, compressor=None, overwrite=True)

    # save lowdim data
    for data_key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
        this_data = list()
        # Check if this key should be loaded from emb files
        # Keys ending with 'emb' are loaded from emb_*.npz files
        use_emb_file = data_key.endswith('emb') and data_key != 'action'
        
        for episode_idx in range(len(episode_ends)):
            ep_len = episode_ends[episode_idx] - episode_starts[episode_idx]
            if ep_len == 0:
                continue
            try:
                if data_key == 'action':
                    # Action always comes from train_data files
                    demo = np.load(f'{dataset_path}/train_data_{episode_idx}.npz')
                    this_data.append(demo[data_key][:ep_len].astype(np.float32))
                elif use_emb_file:
                    # Keys ending with 'emb' come from emb files
                    emb_file_path = f'{dataset_path}/emb_{episode_idx}.npz'
                    if os.path.exists(emb_file_path):
                        demo = np.load(emb_file_path)
                        # emb files have shape (1, T, dim), squeeze first dimension
                        if data_key in demo:
                            emb_data = demo[data_key]
                            if emb_data.ndim == 3 and emb_data.shape[0] == 1:
                                # Shape: (1, T, dim) -> (T, dim)
                                this_data.append(emb_data[0, :ep_len, :].astype(np.float32))
                            else:
                                # Shape: (T, dim) already
                                this_data.append(emb_data[:ep_len, :].astype(np.float32))
                        else:
                            raise KeyError(f"Key {data_key} not found in {emb_file_path}")
                    else:
                        raise FileNotFoundError(f"Emb file not found: {emb_file_path}")
                else:
                    # Regular lowdim keys come from train_data files
                    demo = np.load(f'{dataset_path}/train_data_{episode_idx}.npz')
                    this_data.append(demo[data_key][:ep_len].astype(np.float32))
            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Failed to load {data_key} from episode {episode_idx}: {e}")
                continue
        
        if len(this_data) == 0:
            continue
            
        this_data = np.concatenate(this_data, axis=0)
        
        if data_key == 'action':
            assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape']), \
                f"Action shape mismatch: {this_data.shape} vs {(n_steps,) + tuple(shape_meta['action']['shape'])}"
        else:
            assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][data_key]['shape']), \
                f"Obs {data_key} shape mismatch: {this_data.shape} vs {(n_steps,) + tuple(shape_meta['obs'][data_key]['shape'])}"
        
        _ = data_group.array(
            name=data_key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype
        )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    if len(rgb_keys) > 0:
        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c, h, w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(episode_ends)):
                        ep_len = episode_ends[episode_idx] - episode_starts[episode_idx]
                        if ep_len == 0:
                            continue
                        else:
                            try:
                                demo = np.load(f'{dataset_path}/train_data_{episode_idx}.npz')
                                if key == 'rgb':
                                    hdf5_arr = demo['rgb'][:ep_len,:,:,:3]
                                elif key == 'rgb_wrist':
                                    hdf5_arr = demo['rgb'][:ep_len,:,:,3:]
                                else:
                                    hdf5_arr = demo[key][:ep_len]
                                
                                for hdf5_idx in range(hdf5_arr.shape[0]):
                                    if len(futures) >= max_inflight_tasks:
                                        # limit number of inflight tasks
                                        completed, futures = concurrent.futures.wait(futures,
                                                                                        return_when=concurrent.futures.FIRST_COMPLETED)
                                        for f in completed:
                                            if not f.result():
                                                raise RuntimeError('Failed to encode image!')
                                        pbar.update(len(completed))

                                    zarr_idx = episode_starts[episode_idx] + hdf5_idx
                                    futures.add(
                                        executor.submit(img_copy,
                                                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                            except FileNotFoundError:
                                continue
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
