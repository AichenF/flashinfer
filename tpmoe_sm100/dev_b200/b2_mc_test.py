"""Check torch symmetric memory (gloo world group) gives peer + NVLS multicast addresses on this node."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp


def w(rank, n, path):
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=n)
    import torch.distributed._symmetric_memory as symm_mem
    g = dist.group.WORLD
    t = symm_mem.empty(1 << 20, dtype=torch.uint8, device=f'cuda:{rank}')
    t.zero_()
    h = symm_mem.rendezvous(t, g.group_name)
    print(rank, 'ptr', hex(t.data_ptr()), 'bufs', [hex(p) for p in h.buffer_ptrs], 'mc', hex(h.multicast_ptr), flush=True)
    dist.barrier()


if __name__ == '__main__':
    p = '/tmp/mc_rdv_%d' % os.getpid()
    mp.spawn(w, args=(4, p), nprocs=4)
