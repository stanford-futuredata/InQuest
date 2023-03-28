import numpy as np
import pandas as pd

# DEFINITIONS
NUM_STRATA = 3
MAX_FRAME = 500000

STRATA_STD_SCALE_MIN = 0
STRATA_STD_SCALE_MAX = 9
STRATA_MEAN_RANGES = [(0,3), (3,6), (6,9)]

NUM_SHIFTS_LIST = [1, 2, 3, 4, 5]
DATASETS_PER_SHIFT = 20


def sample_stds(ndim=3):
    # sample vector from unit sphere
    vec = np.random.rand(ndim)
    vec /= np.linalg.norm(vec)

    # sample scale factor
    scale_factor = np.random.uniform(low=STRATA_STD_SCALE_MIN, high=STRATA_STD_SCALE_MAX)

    # return scaled vector
    return vec * scale_factor

def sample_means(ndim=3):
    return np.array([
        np.random.uniform(low=STRATA_MEAN_RANGES[strata_idx][0], high=STRATA_MEAN_RANGES[strata_idx][1])
        for strata_idx in range(ndim)
    ])

def sample_pred_rates(ndim):
    return np.array([np.random.uniform(low=0.0, high=1.0) for _ in range(ndim)])

def compute_samples_per_segment_strata(num_strata, shift_frames):
    samples_per_segment_strata = []
    shift_frames_with_final_frame = shift_frames + [MAX_FRAME]
    for segment_idx, shift_frame in enumerate(shift_frames_with_final_frame):
        total_num_samples = (
            shift_frame
            if segment_idx == 0
            else shift_frame - shift_frames_with_final_frame[segment_idx - 1]
        )
        if shift_frame == MAX_FRAME:
            total_num_samples += 1

        samples_per_segment_strata.append([np.floor((1/num_strata) * total_num_samples) for _ in range(num_strata)])

        strata_idx = 0
        while np.sum(samples_per_segment_strata[segment_idx]) < total_num_samples:
            samples_per_segment_strata[segment_idx][strata_idx] += 1
            strata_idx = (strata_idx + 1) % num_strata
    
    return np.array(samples_per_segment_strata)

def interleave(strata_gt_streams, strata_proxy_streams, strata_pred_streams, segment_idx, samples_per_segment_strata, num_strata):
    gt_stream, proxy_stream, pred_stream = [], [], []
  
    for sample in range(int(np.sum(samples_per_segment_strata[segment_idx]))):
        strata_to_sample = np.random.choice(
            list(range(num_strata)),
            p=samples_per_segment_strata[segment_idx]/np.sum(samples_per_segment_strata[segment_idx])
        )
        while len(strata_gt_streams[strata_to_sample]) == 0:
            strata_to_sample = (strata_to_sample + 1) % num_strata

        gt_sample = strata_gt_streams[strata_to_sample].pop()
        proxy_sample = strata_proxy_streams[strata_to_sample].pop()
        pred_sample = strata_pred_streams[strata_to_sample].pop()

        gt_stream.append(gt_sample)
        proxy_stream.append(proxy_sample)
        pred_stream.append(pred_sample)
    
    return gt_stream, proxy_stream, pred_stream

def create_stream(num_shifts, num_strata, init_stds, init_means, init_pred_rates):
    # compute which frames will have a shift
    shift_frames = sorted(list(np.random.randint(low=1, high=MAX_FRAME - 1, size=num_shifts)))

    # ensure that the same frame isn't sampled twice (by miniscule chance)
    while len(shift_frames) != len(np.unique(shift_frames)):
        shift_frames = sorted(list(np.random.randint(low=1, high=MAX_FRAME - 1, size=num_shifts)))

    # compute samples per shift segment
    samples_per_segment_strata = compute_samples_per_segment_strata(num_strata, shift_frames)

    # initialize strata means, stds., and pred. rates
    strata_means = init_means
    strata_stds = init_stds
    strata_pred_rates = init_pred_rates

    # construct stream
    full_gt_stream, full_proxy_stream, full_pred_stream = [], [], []
    for segment_idx in range(num_shifts + 1):
        strata_gt_streams, strata_proxy_streams, strata_pred_streams = [], [], []

        for strata_idx in range(num_strata):
            # compute groundtruth
            strata_gt_stream = np.random.normal(
                loc=strata_means[strata_idx],
                scale=strata_stds[strata_idx],
                size=(1, int(samples_per_segment_strata[segment_idx][strata_idx])),
            )
            strata_gt_streams.append(list(strata_gt_stream[0]))

            # compute proxies
            proxy_vals = list(strata_gt_stream[0])
            strata_proxy_streams.append(proxy_vals)

            # compute predicates
            predicates = np.random.binomial(
                n=1,
                p=strata_pred_rates[strata_idx],
                size=int(samples_per_segment_strata[segment_idx][strata_idx]),
            )
            predicates = list(map(lambda p : bool(p), predicates))
            strata_pred_streams.append(predicates)

        segment_gt_stream, segment_proxy_stream, segment_pred_stream = interleave(strata_gt_streams, strata_proxy_streams, strata_pred_streams, segment_idx, samples_per_segment_strata, num_strata)
        
        full_gt_stream.extend(segment_gt_stream)
        full_proxy_stream.extend(segment_proxy_stream)
        full_pred_stream.extend(segment_pred_stream)

        # compute new strata means, stds., and pred. rates
        strata_stds = sample_stds(ndim=num_strata)
        strata_means = sample_means(ndim=num_strata)
        strata_pred_rates = sample_pred_rates(ndim=num_strata)
    
    full_proxy_stream = full_proxy_stream - np.min(full_proxy_stream)
    full_proxy_stream = full_proxy_stream / np.max(full_proxy_stream)
    
    return full_gt_stream, full_proxy_stream, full_pred_stream


if __name__ == "__main__":

    for num_shifts in NUM_SHIFTS_LIST:
        for idx in range(DATASETS_PER_SHIFT):
            print(f"NUM_SHIFTS: {num_shifts} -- DATASET: {idx}")
            # sample initial strata stds., means, pred. rates
            random_state = num_shifts * DATASETS_PER_SHIFT + idx
            np.random.seed(random_state)

            init_stds = sample_stds(ndim=NUM_STRATA)
            init_means = sample_means(ndim=NUM_STRATA)
            init_pred_rates = sample_pred_rates(ndim=NUM_STRATA)

            gt_stream, proxy_stream, pred_stream = create_stream(num_shifts, NUM_STRATA, init_stds, init_means, init_pred_rates)

            pd.DataFrame({"car_count": gt_stream, "predicate": pred_stream}).to_csv(f"datasets/final-synthetic/true_num_shifts_{num_shifts}_vec_{idx}.csv", index=False, header=False)
            pd.DataFrame({"proxy": proxy_stream}).to_csv(f"datasets/final-synthetic/prob_num_shifts_{num_shifts}_vec_{idx}.csv", index=False, header=False)
