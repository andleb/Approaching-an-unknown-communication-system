import gc
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch

from clickDetector import clickDetector
from infowavegan import WaveGANGenerator

if __name__ == "__main__":

    genPath = sys.argv[1]
    print(sys.argv)
    print(genPath)

    # PREAMBLE
    nCPU = int(mp.cpu_count())
    nCPU //= 2
    chunksize = 500

    print("nCPU: ", nCPU)
    print("affinity: ", len(os.sched_getaffinity(0)))

    nCat = 5
    Nts = 40
    N = 2500

    print("Param lengths: ", nCat, Nts, N)

    bits = list(range(nCat))
    ts = np.linspace(-12.5, 12.5, Nts)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used :", device)

    batchSize = N
    print("Using gpu batches of size ", batchSize)

    Gi = WaveGANGenerator(slice_len=2 ** 16)
    weights = torch.load(genPath, map_location='cpu')
    Gi.load_state_dict(weights)

    G = torch.nn.DataParallel(Gi)
    G.to(f'cuda:{G.device_ids[0]}')

    _z = torch.FloatTensor(N, 100 - nCat).uniform_(-1, 1).to(device)
    c = torch.FloatTensor(N, nCat).to(device)
    z = torch.cat((c, _z), dim=1)
    z[:, :nCat] = 0

    print("Memory: total, reserved, allocated: ",
          torch.cuda.get_device_properties(0).total_memory,
          torch.cuda.memory_reserved(0),
          torch.cuda.memory_allocated(0)
          )

    print("Opening pool with", nCPU, "CPUs")
    pool = mp.Pool(nCPU)

    # 2-way INTERACTION

    res = []

    # baseline: [00000]
    print("Generating baseline data")

    genData = []
    for zPart in z.split(batchSize, dim=0):
        genData.append(G(zPart).detach().cpu().flatten(start_dim=1, end_dim=2).numpy())

    res.append([pool.map_async(clickDetector, np.concatenate(genData),
                               chunksize=chunksize)])
    del genData
    gc.collect()

    for bit in bits:
        print(f"Generating for bit {bit}, N={N}, {len(ts)} values of t...")

        print("Memory: total, reserved, allocated: ",
              torch.cuda.get_device_properties(0).total_memory,
              torch.cuda.memory_reserved(0),
              torch.cuda.memory_allocated(0)
              )

        z[:, :nCat] = 0

        dummy = []

        for t in ts:

            z[:, bit] = t

            # Workaround due to GPU memory limits
            genData = []
            for zPart in z.split(batchSize, dim=0):
                genData.append(G(zPart).detach().cpu().flatten(start_dim=1, end_dim=2).numpy())

            dummy.append(pool.map_async(clickDetector, np.concatenate(genData),
                                        chunksize=chunksize))

            del genData
            gc.collect()

        print(f"Generated for bit {bit}, N={N}, {len(ts)} values of t...")

        res.append(dummy)
        gc.collect()

    print("Waiting on results")
    # Call in order
    clicks = -1 * np.ones(shape=(len(bits) + 1, len(ts), N), dtype=int)

    for i, bitL in enumerate(res):
        for j, TT in enumerate(bitL):
            result = TT.get()
            clicks[i, j, :] = [el[1] for el in result]
            res[i][j] = [el[0] for el in result]

    pool.close()
    pool.join()

    bits.insert(0, -1)

    print("Saving")
    np.save("data/single" + genPath[7:-5].replace("/", "") + "Bits.npy",
            bits)
    np.save("data/single" + genPath[7:-5].replace("/", "") + "Ts.npy",
            ts)
    np.save("data/single" + genPath[7:-5].replace("/", "") + "Zs.npy",
            z[:, nCat:].detach().cpu())
    np.save("data/single" + genPath[7:-5].replace("/", "") + "Clicks.npy",
            clicks)

    # save the ICIs as hd5
    df1 = pd.DataFrame(res, index=bits, columns=ts)
    df1.to_hdf("data/single" + genPath[7:-5].replace("/", "") + ".hdf5",
               f"{genPath[7:-5].replace('/', '')}")
