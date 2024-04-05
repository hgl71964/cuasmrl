import torch


def do_bench(fn,
             warmup=25,
             rep=100,
             grad_to_none=None,
             quantiles=None,
             fast_flush=True,
             return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

    # Estimate the runtime of the function
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    # for _ in range(5):
    #     cache.zero_()
    #     fn()
    # end_event.record()
    # torch.cuda.synchronize()
    # estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    # n_warmup = max(1, int(warmup / estimate_ms))
    # n_repeat = max(1, int(rep / estimate_ms))

    n_warmup = warmup
    n_repeat = rep

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float)

    return getattr(torch, return_mode)(times).item()
