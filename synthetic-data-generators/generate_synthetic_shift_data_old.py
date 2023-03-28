import numpy as np
import pandas as pd

# DEFINITIONS
NUM_SEGMENTS = 5
NUM_STRATA = 3
MAX_FRAME = 500000
INIT_STRATA_MEANS = [0, 3, 6]
INIT_STRATA_STDS = [1, 1, 1]
SHIFT_STRATA_MEANS = [0, 3, 6]
SHIFT_SEGMENT_IDX = 3
NUM_VEC_SCALES = 5
RANDOM_VECS_PER_SCALE = 10


def sample_spherical(npoints, seed, ndim=3):
    np.random.seed(seed)
    vec = np.random.rand(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def compute_samples_per_stream(num_strata, num_segments):
    samples_per_stream = []
    for segment_idx in range(num_segments):
        total_num_samples = int(MAX_FRAME * 0.1) if segment_idx == 0 else int(MAX_FRAME * 0.9 / 4.0)
        if segment_idx == num_segments - 1:
            total_num_samples += 1
        samples_per_stream.append([np.floor((1/num_strata) * total_num_samples) for _ in range(num_strata)])

        strata_idx = 0
        while np.sum(samples_per_stream[segment_idx]) < total_num_samples:
            samples_per_stream[segment_idx][strata_idx] += 1
            strata_idx = (strata_idx + 1) % num_strata
    
    return np.array(samples_per_stream)

def interleave(strata_gt_streams, strata_proxy_streams, segment_idx, samples_per_stream, num_strata):
    gt_stream, proxy_stream = [], []
  
    for sample in range(int(np.sum(samples_per_stream[segment_idx]))):
        if sample % int(MAX_FRAME * 0.1) == 0:
            print(f"{segment_idx} -- {sample}")

        strata_to_sample = np.random.choice(
            list(range(num_strata)),
            p=samples_per_stream[segment_idx]/np.sum(samples_per_stream[segment_idx])
        )
        while len(strata_gt_streams[strata_to_sample]) == 0:
            strata_to_sample = (strata_to_sample + 1) % num_strata

        gt_sample = strata_gt_streams[strata_to_sample].pop()
        proxy_sample = strata_proxy_streams[strata_to_sample].pop()

        gt_stream.append(gt_sample)
        proxy_stream.append(proxy_sample)
    
    return gt_stream, proxy_stream

def create_stream(shift_idx, num_strata, num_segments):
    # compute samples per stream
    samples_per_stream = compute_samples_per_stream(num_strata, num_segments)

    shift_strata_stds = list(vecs[:,shift_idx])

    # compute groundtruth
    full_gt_stream, full_proxy_stream = [], []
    for segment_idx in range(num_segments):
        strata_gt_streams, strata_proxy_streams = [], []
        strata_means = INIT_STRATA_MEANS if segment_idx < SHIFT_SEGMENT_IDX else SHIFT_STRATA_MEANS
        strata_stds = INIT_STRATA_STDS if segment_idx < SHIFT_SEGMENT_IDX else shift_strata_stds
        for strata_idx in range(num_strata):
            strata_gt_stream = np.random.normal(
                loc=strata_means[strata_idx],
                scale=strata_stds[strata_idx],
                size=(1, int(samples_per_stream[segment_idx][strata_idx])),
            )
            strata_gt_streams.append(list(strata_gt_stream[0]))

            proxy_vals = list(strata_gt_stream[0])
            strata_proxy_streams.append(list(proxy_vals))

        segment_gt_stream, segment_proxy_stream = interleave(strata_gt_streams, strata_proxy_streams, segment_idx, samples_per_stream, num_strata)
        
        full_gt_stream.extend(segment_gt_stream)
        full_proxy_stream.extend(segment_proxy_stream)
    
    full_proxy_stream = full_proxy_stream - np.min(full_proxy_stream)
    full_proxy_stream = full_proxy_stream / np.max(full_proxy_stream)
    
    return full_gt_stream, full_proxy_stream


if __name__ == "__main__":
    # compute and scale vecs
    vecs = sample_spherical(NUM_VEC_SCALES * RANDOM_VECS_PER_SCALE, seed=12345)
    for idx in range(NUM_VEC_SCALES):
        start_col = idx * RANDOM_VECS_PER_SCALE
        end_col = (idx + 1) * RANDOM_VECS_PER_SCALE
        vecs[:, start_col:end_col] = vecs[:, start_col:end_col] * (idx+1) * np.sqrt(3)

    for shift_idx in range(NUM_VEC_SCALES * RANDOM_VECS_PER_SCALE):
        gt_stream, proxy_stream = create_stream(shift_idx, NUM_STRATA, NUM_SEGMENTS)

        scale_multiple = (shift_idx // 10) + 1
        vec_idx = (shift_idx % 10) + 1

        pd.DataFrame({"car_count": gt_stream}).to_csv(f"datasets/final-synthetic-old/true_scale_{scale_multiple}_vec_{vec_idx}.csv", index=False, header=False)
        pd.DataFrame({"proxy": proxy_stream}).to_csv(f"datasets/final-synthetic-old/prob_scale_{scale_multiple}_vec_{vec_idx}.csv", index=False, header=False)
