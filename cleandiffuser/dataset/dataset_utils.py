from typing import Dict, Callable, List, Union
import numpy as np
import torch
import numba
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
import cleandiffuser.dataset.rotation_conversions as rc
import functools
from tqdm import tqdm


# -----------------------------------------------------------------------------#
# ------------------------------ SequenceSampler ------------------------------#
# -----------------------------------------------------------------------------#

# Original implemetation: https://github.com/real-stanford/diffusion_policy
# Observation Horizon: To|n_obs_steps
# Action Horizon: Ta|n_action_steps
# Prediction Horizon: T|horizon
# To = 3
# Ta = 4
# T = 6
# |o|o|o|
# | | |a|a|a|a|
# pad_before = 2
# pad_after = 3

@numba.jit(nopython=True)
def create_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0, 
        pad_after: int = 0,
        debug: bool = True) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert (start_offset >= 0)
                assert (end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            sequence_length: int,
            pad_before: int = 0,
            pad_after: int = 0,
            keys=None,
            key_first_k=dict(),
            zero_padding: bool = False,
    ):
        """
            key_first_k: dict str: int
                Only take first k data from these keys (to improve perf)
        """
        super().__init__()
        assert (sequence_length >= 1)

        # all keys
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]

        # create indices
        # indices (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # buffer_start_idx and buffer_end_idx define the actual start and end positions of the sample sequence within the original dataset.
        # sample_start_idx and sample_end_idx define the relative start and end positions within the sample sequence, 
        # which is particularly useful when dealing with padding as it can affect the actual length of the sequence.
        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.zero_padding = zero_padding
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


class LongMemSequenceSampler:
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            sequence_length: int,
            pad_before: int = 0,
            pad_after: int = 0,
            keys=None,
            key_first_k=dict(),
            zero_padding: bool = False,
            memory_type: str = 'random_from_full',
            subsample_ratio: int = 10,
            maximum_memory_length: int = 16,
    ):
        """
            key_first_k: dict str: int
                Only take first k data from these keys (to improve perf)
            subsample_ratio: int
                Ratio for subsampling memory frames. Samples every N frames from 0 
                to the current index. This means earlier frames have fewer samples, later frames have more.
        """
        super().__init__()
        assert (sequence_length >= 1)

        # all keys
        if keys is None:
            keys = list(replay_buffer.keys())

        self.episode_ends = replay_buffer.episode_ends[:]

        # create indices
        # indices (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # buffer_start_idx and buffer_end_idx define the actual start and end positions of the sample sequence within the original dataset.
        # sample_start_idx and sample_end_idx define the relative start and end positions within the sample sequence, 
        # which is particularly useful when dealing with padding as it can affect the actual length of the sequence.
        indices = create_indices(
            episode_ends=self.episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.zero_padding = zero_padding
        self.key_first_k = key_first_k
        self.keys_need_mem = list(keys)
        self.keys_need_mem.remove('action')
        self.memory_type = memory_type
        self.subsample_ratio = subsample_ratio
        self.maximum_memory_length = maximum_memory_length

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        # find last episode start idx
        episode_num_idx = np.searchsorted(self.episode_ends, buffer_start_idx) - 1
        episode_num_idx = episode_num_idx
        episode_start_idx = self.episode_ends[episode_num_idx] if episode_num_idx >= 0 else 0
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        # sample memory using subsample_ratio
        seq_begin_index = buffer_start_idx - episode_start_idx
        for key in self.keys_need_mem:
            input_arr = self.replay_buffer[key]
            
            # Sample randomly from each segment of subsample_ratio frames
            # Divide [0, seq_begin_index) into segments of size subsample_ratio
            # and randomly sample one frame from each segment
            if seq_begin_index == 0:
                selected_frame_idx = [episode_start_idx]
            else:
                # Calculate number of segments
                num_segments = (seq_begin_index + self.subsample_ratio - 1) // self.subsample_ratio
                
                # Vectorized: generate all segment starts
                seg_starts = np.arange(num_segments) * self.subsample_ratio
                # Vectorized: generate all segment ends (cap at seq_begin_index)
                seg_ends = np.minimum(seg_starts + self.subsample_ratio, seq_begin_index)
                
                # Filter out segments that are out of range
                valid_mask = seg_starts < seq_begin_index
                seg_starts = seg_starts[valid_mask]
                seg_ends = seg_ends[valid_mask]
                
                # Vectorized: generate random indices for each segment
                segment_sizes = seg_ends - seg_starts
                # Ensure segment_sizes are at least 1 to avoid np.random.randint(0, 0)
                segment_sizes[segment_sizes == 0] = 1
                random_offsets = np.random.randint(0, segment_sizes, size=len(seg_starts))
                selected_frame_idx = (seg_starts + random_offsets + episode_start_idx).tolist()
            
            # Ensure at least one frame
            if len(selected_frame_idx) == 0:
                selected_frame_idx = [episode_start_idx]
            
            mem_data = input_arr[selected_frame_idx]
            
            # truncate memory to maximum_memory_length
            if len(mem_data) > self.maximum_memory_length:
                mem_data = mem_data[-self.maximum_memory_length:]
                selected_frame_idx = selected_frame_idx[-self.maximum_memory_length:]
            
            result[f'{key}_mem'] = mem_data
            result['ep_step'] = np.array(selected_frame_idx, dtype=np.float32) - np.ones(len(selected_frame_idx), dtype=np.float32) * episode_start_idx
        
        # Store actual_num_frames for dynamic batching
        result['actual_num_frames'] = len(selected_frame_idx) if len(self.keys_need_mem) > 0 else 1
        
        return result

# -----------------------------------------------------------------------------#
# ---------------------------- Rotation Transformer ---------------------------#
# -----------------------------------------------------------------------------#

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
                 from_rep='axis_angle',
                 to_rep='rotation_6d',
                 from_convention=None,
                 to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(rc, f'{from_rep}_to_matrix'),
                getattr(rc, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(rc, f'matrix_to_{to_rep}'),
                getattr(rc, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.tensor(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


# -----------------------------------------------------------------------------#
# --------------------------- multi-field normalizer --------------------------#
# -----------------------------------------------------------------------------#

def empirical_cdf(sample):
    """ 
    Compute empirical CDF using torch (torch implementation)
    https://stackoverflow.com/a/33346366 
    """
    # Convert to torch if numpy
    if isinstance(sample, np.ndarray):
        sample = torch.from_numpy(sample.astype(np.float32))
    else:
        sample = sample.float()
    
    # Find unique values and their counts using torch
    unique_vals, inverse_indices, counts = torch.unique(
        sample, return_inverse=True, return_counts=True
    )
    
    # Compute cumulative probabilities
    cumprob = torch.cumsum(counts.float(), dim=0) / sample.numel()
    
    return unique_vals, cumprob


class CDFNormalizer1d:
    """
        CDF normalizer for a single dimension (torch implementation)
    """

    def __init__(self, X):
        # Convert to torch if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        else:
            X = X.float()
        
        assert X.ndim == 1
        self.X = X
        
        quantiles, cumprob = empirical_cdf(self.X)
        
        # Store quantiles and cumprob for interpolation
        self.quantiles = quantiles
        self.cumprob = cumprob
        self.xmin = torch.min(quantiles)
        self.xmax = torch.max(quantiles)
        self.ymin = torch.min(cumprob)
        self.ymax = torch.max(cumprob)
        
        # Store as numpy for backward compatibility
        self.quantiles_np = quantiles.numpy() if isinstance(X, torch.Tensor) else quantiles
        self.cumprob_np = cumprob.numpy() if isinstance(X, torch.Tensor) else cumprob
        self.xmin_np = self.xmin.item() if isinstance(X, torch.Tensor) else self.xmin
        self.xmax_np = self.xmax.item() if isinstance(X, torch.Tensor) else self.xmax
        self.ymin_np = self.ymin.item() if isinstance(X, torch.Tensor) else self.ymin
        self.ymax_np = self.ymax.item() if isinstance(X, torch.Tensor) else self.ymax

    def _interp1d(self, x, xp, fp):
        """
        Linear 1D interpolation using torch (similar to numpy.interp)
        """
        # Handle scalar input
        if x.dim() == 0:
            x = x.unsqueeze(0)
            was_scalar = True
        else:
            was_scalar = False
        
        # Find the indices where x should be inserted to maintain sorted order
        indices = torch.searchsorted(xp, x, right=False)
        
        # Handle boundary cases
        indices = torch.clamp(indices, 0, len(xp) - 1)
        indices_prev = torch.clamp(indices - 1, 0, len(xp) - 1)
        
        # Get values at indices
        xp_prev = xp[indices_prev]
        xp_next = xp[indices]
        fp_prev = fp[indices_prev]
        fp_next = fp[indices]
        
        # Linear interpolation
        # Handle exact matches and zero denominators
        exact_match = (xp_next == xp_prev)
        denom = xp_next - xp_prev
        # Avoid division by zero - use fp_prev when exact match
        alpha = torch.where(
            exact_match,
            torch.zeros_like(denom),
            (x - xp_prev) / torch.where(denom == 0, torch.ones_like(denom), denom)
        )
        result = torch.where(exact_match, fp_prev, fp_prev + alpha * (fp_next - fp_prev))
        
        if was_scalar:
            result = result.squeeze(0)
        
        return result

    def normalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Clip and interpolate
        x = torch.clamp(x, self.xmin, self.xmax)
        y = self._interp1d(x, self.quantiles, self.cumprob)
        y = 2 * y - 1
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            y = y.numpy()
        return y

    def unnormalize(self, x, eps=1e-4):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = (x + 1) / 2.0
        
        # Check for out of range values
        if torch.any(x < self.ymin - eps) or torch.any(x > self.ymax + eps):
            x_min_val = torch.min(x).item() if isinstance(x, torch.Tensor) else x.min()
            x_max_val = torch.max(x).item() if isinstance(x, torch.Tensor) else x.max()
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x_min_val}, {x_max_val}] | '''
                f'''x : [{self.xmin_np}, {self.xmax_np}] | '''
                f'''y: [{self.ymin_np}, {self.ymax_np}]''')
        
        x = torch.clamp(x, self.ymin, self.ymax)
        y = self._interp1d(x, self.cumprob, self.quantiles)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            y = y.numpy()
        return y


class CDFNormalizer:
    """
        makes training data uniform (over each dimension) by transforming it with marginal CDFs (torch implementation)
    """

    def __init__(self, X):
        # Convert to torch if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        else:
            X = X.float()
        
        self.X = X
        self.mins = torch.min(X, dim=0)[0]
        self.maxs = torch.max(X, dim=0)[0]
        self.dim = X.shape[-1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)]
        
        # Store as numpy for backward compatibility
        self.mins_np = self.mins.numpy() if isinstance(X, torch.Tensor) else self.mins
        self.maxs_np = self.maxs.numpy() if isinstance(X, torch.Tensor) else self.maxs

    def wrap(self, fn_name, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        shape = x.shape
        x = x.reshape(-1, self.dim)
        
        # Process each dimension
        out_list = []
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            result = fn(x[:, i])
            # Ensure result is torch tensor
            if isinstance(result, np.ndarray):
                result = torch.from_numpy(result)
            out_list.append(result)
        
        # Stack results
        out = torch.stack(out_list, dim=1)
        out = out.reshape(shape)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            out = out.numpy()
        return out

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)


class GaussianNormalizer:
    """
        normalizes data to have zero mean and unit variance (torch implementation)
    """

    def __init__(self, X):
        # Convert to torch tensor if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        else:
            X = X.float()
        
        # Compute statistics using torch
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0)
        self.stds = torch.where(self.stds == 0, torch.ones_like(self.stds), self.stds)
        
        # Store as numpy for backward compatibility if needed
        self.means_np = self.means.numpy() if isinstance(X, torch.Tensor) else self.means
        self.stds_np = self.stds.numpy() if isinstance(X, torch.Tensor) else self.stds

    def normalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        
        # Normalize using torch
        result = (x - self.means) / self.stds
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.numpy()
        return result

    def unnormalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        
        # Unnormalize using torch
        result = x * self.stds + self.means
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.numpy()
        return result


class ImageNormalizer:
    """
        normalizes image data from range [0, 1] to [-1, 1] (torch implementation).
    """

    def __init__(self, device='cpu'):
        self.device = device

    def normalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        result = x * 2.0 - 1.0
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.numpy()
        return result

    def unnormalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        result = (x + 1.0) / 2.0
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.numpy()
        return result


class MinMaxNormalizer:
    """
        normalizes data through maximum and minimum expansion (torch implementation).
    """

    def __init__(self, X, device=None):
        # Convert to torch tensor if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        else:
            X = X.float()
        
        # Move to device if specified
        if device is not None:
            X = X.to(device)
        
        # Reshape and compute statistics using torch
        X = X.reshape(-1, X.shape[-1])
        self.min = torch.min(X, dim=0)[0]
        self.max = torch.max(X, dim=0)[0]
        self.range = self.max - self.min
        
        # Handle zero range
        zero_mask = self.range == 0
        if torch.any(zero_mask):
            print("Warning: Some features have the same min and max value. These will be set to 0.")
            self.range = torch.where(zero_mask, torch.ones_like(self.range), self.range)
        
        # Store device for later use
        self.device = device if device is not None else X.device
        
        # Store as numpy for backward compatibility if needed
        self.min_np = self.min.cpu().numpy() if isinstance(X, torch.Tensor) else self.min
        self.max_np = self.max.cpu().numpy() if isinstance(X, torch.Tensor) else self.max
        self.range_np = self.range.cpu().numpy() if isinstance(X, torch.Tensor) else self.range

    def normalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Move to same device as normalizer parameters
        x = x.to(self.device)
        min_val = self.min.to(self.device)
        range_val = self.range.to(self.device)
        
        # Normalize to [0, 1] then to [-1, 1]
        nx = (x - min_val) / range_val
        nx = nx * 2.0 - 1.0
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            nx = nx.cpu().numpy()
        return nx

    def unnormalize(self, x):
        # Support both numpy and torch inputs
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Move to same device as normalizer parameters
        x = x.to(self.device)
        min_val = self.min.to(self.device)
        range_val = self.range.to(self.device)
        
        # Unnormalize from [-1, 1] to [0, 1] then to original range
        nx = (x + 1.0) / 2.0
        result = nx * range_val + min_val
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.cpu().numpy()
        return result


class EmptyNormalizer:
    """
        do nothing and change nothing
    """

    def __init__(self):
        pass

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


# -----------------------------------------------------------------------------#
# ------------------------------- useful tool ---------------------------------#
# -----------------------------------------------------------------------------#

def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b


class SortedByLengthSampler(torch.utils.data.Sampler):
    """
    Sampler that sorts dataset indices by actual_num_frames (memory length).
    This ensures that samples with similar memory lengths are grouped together,
    which is more efficient for dynamic batching.
    """
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Pre-compute actual_num_frames for all samples
        self.actual_num_frames_list = []
        
        # Check if dataset has a faster method to compute actual_num_frames
        # without loading all data
        if hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'indices'):
            # Fast path: compute directly from sampler without loading data
            sampler = dataset.sampler
            episode_ends = sampler.episode_ends
            for idx in tqdm(range(len(dataset)), desc="Computing actual_num_frames for all samples"):
                buffer_start_idx = sampler.indices[idx][0]
                # Find episode start index
                episode_num_idx = np.searchsorted(episode_ends, buffer_start_idx) - 1
                episode_start_idx = episode_ends[episode_num_idx] if episode_num_idx >= 0 else 0
                # Calculate seq_begin_index (same logic as in sample_sequence)
                seq_begin_index = buffer_start_idx - episode_start_idx
                # Calculate actual_num_frames using subsample_ratio (same logic as in sample_sequence)
                if seq_begin_index == 0:
                    actual_num_frames = 1
                else:
                    num_segments = (seq_begin_index + sampler.subsample_ratio - 1) // sampler.subsample_ratio
                    actual_num_frames = num_segments if num_segments > 0 else 1
                maximum_memory_length = sampler.maximum_memory_length
                self.actual_num_frames_list.append(min(actual_num_frames, maximum_memory_length))
        else:
            # Fallback: use dataset.__getitem__ (slow but works for any dataset)
            for idx in tqdm(range(len(dataset)), desc="Computing actual_num_frames for all samples"):
                sample = dataset[idx]
                actual_num_frames = int(sample['obs']['actual_num_frames'])
                maximum_memory_length = sampler.maximum_memory_length
                # else:
                #     # Fallback: calculate from ep_step length
                #     if 'obs' in sample and 'ep_step' in sample['obs']:
                #         actual_num_frames = len(sample['obs']['ep_step'])
                #     else:
                #         actual_num_frames = 1
                self.actual_num_frames_list.append(min(actual_num_frames, maximum_memory_length))
        
        # Sort indices by actual_num_frames
        # argsort returns indices that would sort the array
        self.sorted_indices = np.argsort(self.actual_num_frames_list)
        print(f"Sorted {len(self.sorted_indices)} samples by memory length")
        print(f"Memory length range: {min(self.actual_num_frames_list)} - {max(self.actual_num_frames_list)}")
    
    def __iter__(self):
        if self.shuffle:
            # Shuffle within groups of similar lengths, but also shuffle group order
            # Group indices by similar actual_num_frames
            groups = {}
            for original_idx in self.sorted_indices:
                # sorted_indices contains the original dataset indices in sorted order
                num_frames = self.actual_num_frames_list[original_idx]
                # Round to nearest 5 for grouping (you can adjust this)
                group_key = (num_frames // 5) * 5
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(original_idx)
            
            # Shuffle within each group
            for group_key in groups:
                np.random.shuffle(groups[group_key])
            
            # Shuffle the order of groups (not sorted by group_key)
            group_keys = list(groups.keys())
            np.random.shuffle(group_keys)
            
            # Concatenate groups in shuffled order
            shuffled_indices = []
            for group_key in group_keys:
                shuffled_indices.extend(groups[group_key])
            
            return iter(shuffled_indices)
        else:
            # Return sorted indices (which are the original dataset indices)
            return iter(self.sorted_indices)
    
    def __len__(self):
        return len(self.sorted_indices)


def dynamic_batch_collate_fn(batch):
    """
    Optimized collate function for dynamic batching with variable memory lengths.
    
    Since samples are already sorted by length via SortedByLengthSampler,
    we just need to pad all samples in the batch to the same length (max length).
    This function is optimized to minimize CPU operations and prepare data for GPU.
    
    Args:
        batch: List of samples from the dataset (already sorted by length)
        
    Returns:
        Collated batch with padded memory sequences, ready for GPU transfer
    """
    # Extract actual_num_frames for each sample to find max length
    # Optimize: do this in a single pass
    max_len = 1
    for sample in batch:
        if 'obs' in sample and 'actual_num_frames' in sample['obs']:
            max_len = max(max_len, int(sample['obs']['actual_num_frames']))
        elif 'obs' in sample and 'ep_step' in sample['obs']:
            max_len = max(max_len, len(sample['obs']['ep_step']))
    
    # Pre-allocate lists for faster appending
    padded_samples = []
    
    # Process all samples
    for sample in batch:
        padded_sample = {}
        
        # Copy non-memory keys as-is
        for key, value in sample.items():
            if key == 'obs':
                padded_sample[key] = {}
                for obs_key, obs_value in value.items():
                    if obs_key.endswith('_mem'):
                        # Memory key: convert to tensor and pad
                        # Optimize: convert numpy to tensor in one step
                        if isinstance(obs_value, np.ndarray):
                            # Use pin_memory-friendly conversion
                            obs_value = torch.from_numpy(obs_value).contiguous()
                        elif not isinstance(obs_value, torch.Tensor):
                            obs_value = torch.tensor(obs_value)
                        else:
                            obs_value = obs_value.contiguous()
                        
                        current_len = obs_value.shape[0]
                        if current_len < max_len:
                            # Pad with the first frame at the beginning
                            first_frame = obs_value[0:1]  # Keep the first dimension
                            pad_len = max_len - current_len
                            # Repeat first frame pad_len times along the first dimension
                            pad = first_frame.repeat(pad_len, *([1] * (len(obs_value.shape) - 1)))
                            padded_value = torch.cat([pad, obs_value], dim=0)
                        else:
                            padded_value = obs_value
                        padded_sample[key][obs_key] = padded_value
                    elif obs_key == 'ep_step':
                        # Pad ep_step similarly
                        if isinstance(obs_value, np.ndarray):
                            obs_value = torch.from_numpy(obs_value).contiguous()
                        elif not isinstance(obs_value, torch.Tensor):
                            obs_value = torch.tensor(obs_value)
                        else:
                            obs_value = obs_value.contiguous()
                        
                        current_len = len(obs_value)
                        if current_len < max_len:
                            # Pad with zeros at the beginning
                            pad = torch.zeros((max_len - current_len,), dtype=obs_value.dtype)
                            padded_value = torch.cat([pad, obs_value], dim=0)
                        else:
                            padded_value = obs_value
                        padded_sample[key][obs_key] = padded_value
                    elif obs_key == 'actual_num_frames':
                        # Store original length for potential use
                        padded_sample[key][obs_key] = obs_value
                    else:
                        # Non-memory key, copy as-is
                        # Convert numpy to tensor if needed for consistency
                        if isinstance(obs_value, np.ndarray):
                            obs_value = torch.from_numpy(obs_value).contiguous()
                        padded_sample[key][obs_key] = obs_value
            else:
                # Non-obs key, copy as-is
                # Convert numpy to tensor if needed
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).contiguous()
                padded_sample[key] = value
        
        padded_samples.append(padded_sample)
    
    # Use default collate for the padded batch
    # This will handle batching and pin_memory if enabled
    return torch.utils.data.dataloader.default_collate(padded_samples)

def loop_two_dataloaders(dl1, dl2):
    while True:
        for b1, b2 in zip(dl1, dl2):
            yield b1, b2