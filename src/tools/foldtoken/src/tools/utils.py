import torch
import numpy as np
from collections.abc import Mapping, Sequence
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import set_loky_pickler
# from src.tools.affine_utils import Rigid

def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        if isinstance(obj, str):
            return obj
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
    elif isinstance(obj, Rigid):
        return obj.to(*args, **kwargs)
    else:
        return obj
        

    raise TypeError("Can't transfer object type `%s`" % type(obj))

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """

    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    # n_jobs = 60
    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
    )

    return results



def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval


class RectifiedFlow():
  def __init__(self, model=None, num_steps=1000):
    self.model = model
    self.N = num_steps

  def get_train_tuple(self, z0=None, z1=None, t=None, batch_id=None):
    dtype = z0.dtype
    if batch_id is None:
        t = torch.rand((z1.shape[0], 1, 1), device=z0.device, dtype=dtype)
    else:
        t = torch.rand((batch_id.unique().shape[0], 1, 1), device=z0.device, dtype=dtype)
        t = t[batch_id]
    z_t =  t * z1 + (1.-t) * z0
    # target = z1 - z0
    target = z1 - z_t

    return z_t, t, target

  @torch.no_grad()
  def sample_ode(self, z0=None,  N=None, batch_id=None, chain_encoding=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    norm_max = torch.ones_like(batch_id, device=z0.device)[:,None] * 10.

    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1,1), device=z0.device) * i / N
      z1_hat, all_preds, vq_los = self.model(chain_encoding, batch_id, z, t, norm_max)
      z = z.detach().clone() + (z1_hat/norm_max[...,None]-z)/(1-t) * dt
    #   z =  t * z1_hat + (1.-t) * z0

      traj.append(z.detach().clone()*norm_max[...,None])

    return traj