# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/mosaicml/composer/blob/f2a2dc820/composer/callbacks/speed_monitor.py
from collections import deque
from typing import Deque, Dict, Optional, Union

from lightning.fabric.utilities.throughput import _MonotonicWindow

_THROUGHPUT_METRICS = Dict[str, Union[int, float]]


# The API design of this class follows `torchmetrics.Metric` but it doesn't need to be an actual Metric because there's
# no need for synchronization or reduction as it doesn't use Tensors at all.
class Throughput:
    """Computes throughput.

    +------------------------+-------------------------------------------------------------------------------------+
    | Key                    | Value                                                                               |
    +========================+=====================================================================================+
    | batches_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of batches |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of samples |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | tokens_per_sec         | Rolling average (over ``window_size`` most recent updates) of the number of tokens  |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | flpps_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of flops   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/batches_per_sec | batches_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/samples_per_sec | samples_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/tokens_per_sec  | items_per_sec divided by world size. This may include padding depending on the data |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/flops_per_sec   | flops_per_sec divided by world size.                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/mfu             | device/flops_per_sec divided by world size.                                         |
    +--------------------------+-----------------------------------------------------------------------------------+
    | time                   | Total elapsed time                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | batches                | Total batches seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples                | Total samples seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | lengths                | Total items seen                                                                    |
    +--------------------------+-----------------------------------------------------------------------------------+

    Example::

        throughput = Throughput()
        t0 = time()
        for i in range(1000):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, samples=i)
            if i % 10 == 0:
                print(throughput.compute())

    Notes:
        - The implementation assumes that devices FLOPs are all the same as it normalizes by the world size and only
          takes a single ``available_flops`` value.
        - tokens_per_sec, flops_per_sec and MFU do not account for padding if present.
        - non_padding_tokens_per_sec accounts for padding.

    Args:
        available_flops: Number of theoretical flops available for a single device.
        world_size: Number of devices available across hosts. Global metrics are not included if the world size is 1.
        window_size: Number of batches to use for a rolling average.
        separator: Key separator to use when creating per-device and global metrics.


    TODO: understand differnece between device-specific and global metrics
    """

    def __init__(
        self,
        available_flops: Optional[float] = None,
        world_size: int = 1,
        window_size: int = 100,
        separator: str = "/",
    ) -> None:
        self.available_flops = available_flops
        self.separator = separator
        assert world_size > 0
        self.world_size = world_size

        # throughput is computed over a window of values. at least 2 is enforced since it looks at the difference
        # between the first and last elements
        assert window_size > 1
        # custom class instead of `deque(maxlen=)` because it's easy for users to mess up their timer/counters and log
        # values that do not increase monotonically. this class will raise an error if that happens.
        self._time: _MonotonicWindow[float] = _MonotonicWindow(maxlen=window_size)
        self._batches: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._samples: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._proteins: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._lengths: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._non_padding_lengths: _MonotonicWindow[int] = _MonotonicWindow(
            maxlen=window_size
        )
        self._flops: Deque[int] = deque(maxlen=window_size)

    def update(
        self,
        *,
        time: float,
        batches: int,
        samples: int,
        lengths: Optional[int] = None,
        non_padding_lengths: Optional[int] = None,
        proteins: Optional[int] = None,
        flops: Optional[int] = None,
    ) -> None:
        """Update throughput metrics.

        Args:
            time: Total elapsed time in seconds. It should monotonically increase by the iteration time with each
                call.
            batches: Total batches seen per device. It should monotonically increase with each call.
            samples: Total samples (documents) seen per device. It should monotonically increase by the batch size with each call.
            lengths: Total length (in tokens) of the samples seen. It should monotonically increase by the lengths of a batch with
                each call.
            proteins: Total distinct proteins seen (i.e. number of sep tokens).
            flops: Flops elapased per device since last ``update()`` call. You can easily compute this by using
                :func:`measure_flops` and multiplying it by the number of batches that have been processed.
                The value might be different in each device if the batch size is not the same.

        """
        self._time.append(time)
        self._batches.append(batches)
        self._samples.append(samples)
        if proteins is not None:
            if proteins < samples:
                raise ValueError(
                    f"Expected sequences ({proteins}) to be greater or equal than samples ({samples})"
                )
            self._proteins.append(proteins)
            if len(self._samples) != len(self._proteins):
                raise RuntimeError(
                    f"If proteins are passed ({len(self._proteins)}), there needs to be the same number of samples"
                    f" ({len(self._samples)})"
                )
        if lengths is not None:
            if lengths < samples:
                raise ValueError(
                    f"Expected lengths ({lengths}) to be greater or equal than samples ({samples})"
                )
            self._lengths.append(lengths)
            if len(self._samples) != len(self._lengths):
                raise RuntimeError(
                    f"If lengths are passed ({len(self._lengths)}), there needs to be the same number of samples"
                    f" ({len(self._samples)})"
                )
        if non_padding_lengths is not None:
            if non_padding_lengths < samples:
                raise ValueError(
                    f"Expected non_padding_lengths ({non_padding_lengths}) to be greater or equal than samples"
                    f" ({samples})"
                )
            self._non_padding_lengths.append(non_padding_lengths)
            if len(self._samples) != len(self._non_padding_lengths):
                raise RuntimeError(
                    f"If non_padding_lengths are passed ({len(self._non_padding_lengths)}), there needs to be the same"
                    f" number of samples ({len(self._samples)})"
                )
        if flops is not None:
            # sum of flops across ranks
            self._flops.append(flops * self.world_size)

    def compute(self) -> _THROUGHPUT_METRICS:
        """Compute throughput metrics."""
        metrics = {
            "time": self._time[-1],
            "batches": self._batches[-1],
            "documents": self._samples[-1],
        }
        if self._proteins:
            metrics["proteins"] = self._proteins[-1]
        if self._lengths:
            metrics["tokens"] = self._lengths[-1]
        if self._non_padding_lengths:
            metrics["non_padding_tokens"] = self._non_padding_lengths[-1]

        # add_global_metrics = self.world_size > 1
        add_global_metrics = True
        # a different but valid design choice would be to still compute all these metrics even if the window of values
        # has not been filled
        if len(self._time) == self._time.maxlen:
            elapsed_time = self._time[-1] - self._time[0]
            elapsed_batches = self._batches[-1] - self._batches[0]
            elapsed_samples = self._samples[-1] - self._samples[0]
            # we are safe from ZeroDivisionError thanks to `_MonotonicWindow`
            dev_samples_per_sec = elapsed_samples / elapsed_time
            dev_batches_per_sec = elapsed_batches / elapsed_time
            metrics.update(
                {
                    f"device{self.separator}batches_per_sec": elapsed_batches
                    / elapsed_time,
                    f"device{self.separator}documents_per_sec": dev_samples_per_sec,
                }
            )
            if add_global_metrics:
                samples_per_sec = dev_batches_per_sec * self.world_size
                metrics.update(
                    {
                        "batches_per_sec": samples_per_sec,
                        "documents_per_sec": dev_samples_per_sec * self.world_size,
                    }
                )

            if len(self._proteins) == self._proteins.maxlen:
                elapsed_proteins = self._proteins[-1] - self._proteins[0]
                dev_proteins_per_sec = elapsed_proteins / elapsed_time
                metrics[
                    f"device{self.separator}proteins_per_sec"
                ] = dev_proteins_per_sec
                if add_global_metrics:
                    proteins_per_sec = dev_proteins_per_sec * self.world_size
                    metrics["proteins_per_sec"] = proteins_per_sec

            if len(self._lengths) == self._lengths.maxlen:
                elapsed_lengths = self._lengths[-1] - self._lengths[0]
                dev_items_per_sec = elapsed_lengths / elapsed_time
                metrics[f"device{self.separator}tokens_per_sec"] = dev_items_per_sec
                if add_global_metrics:
                    items_per_sec = dev_items_per_sec * self.world_size
                    metrics["tokens_per_sec"] = items_per_sec

            if len(self._non_padding_lengths) == self._non_padding_lengths.maxlen:
                elapsed_non_padding_lengths = (
                    self._non_padding_lengths[-1] - self._non_padding_lengths[0]
                )
                dev_items_per_sec = elapsed_non_padding_lengths / elapsed_time
                metrics[
                    f"device{self.separator}non_padding_tokens_per_sec"
                ] = dev_items_per_sec
                if add_global_metrics:
                    items_per_sec = dev_items_per_sec * self.world_size
                    metrics["non_padding_tokens_per_sec"] = items_per_sec

        if len(self._flops) == self._flops.maxlen:
            elapsed_flops = sum(self._flops) - self._flops[0]
            elapsed_time = self._time[-1] - self._time[0]
            flops_per_sec = elapsed_flops / elapsed_time
            dev_flops_per_sec = flops_per_sec / self.world_size
            if add_global_metrics:
                metrics["flops_per_sec"] = flops_per_sec
            metrics[f"device{self.separator}flops_per_sec"] = dev_flops_per_sec
            if self.available_flops:
                metrics[f"device{self.separator}mfu"] = (
                    dev_flops_per_sec / self.available_flops
                )

        return metrics

    def reset(self) -> None:
        self._time.clear()
        self._batches.clear()
        self._samples.clear()
        self._lengths.clear()
        self._non_padding_lengths.clear()
        self._flops.clear()
