import os
# NOTE: set GPU thing here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from fid_score import _compute_statistics_of_path
from inception import InceptionV3

# load sample and specify output path
sample_path = '../../fid_stats/FairFace/unbiased_all_race_samples.npz'
output_path = '../../fid_stats/FairFace/unbiased_all_race_fid_stats.npz'

cuda = True
dims = 2048
batch_size = 100

if not os.path.exists(sample_path):
    raise RuntimeError('Invalid path: %s' % sample_path)

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

model = InceptionV3([block_idx])
if cuda:
    model.cuda()
    
print("calculate FID stats..", end=" ", flush=True)

mu, sigma = _compute_statistics_of_path(sample_path, model, batch_size, dims, cuda)
np.savez_compressed(output_path, mu=mu, sigma=sigma)

print("finished saving pre-computed statistics to: {}".format(output_path))