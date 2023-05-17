import numpy as np
import pandas as pd

# DEFINITIONS
NUM_STRATA = 3
BETA = 0.75
MAX_FRAMES = 500000


def compute_samples_per_segment_stratum(shift_indices, num_strata):
    samples_per_segment_stratum = []
    for segment_idx in range(len(shift_indices) + 1):
        # compute samples in segment and split them evenly across strata;
        # later we will use random p_tk's to determine the subset of predicate matching samples
        total_num_samples = 0
        if segment_idx == 0:
            total_num_samples = shift_indices[segment_idx]
        elif segment_idx == len(shift_indices):
            total_num_samples = (MAX_FRAMES + 1) - shift_indices[segment_idx-1]
        else:
            total_num_samples = shift_indices[segment_idx] - shift_indices[segment_idx - 1]

        samples_per_segment_stratum.append([np.floor((1/num_strata) * total_num_samples) for _ in range(num_strata)])

        # add back any rounded off samples
        strata_idx = 0
        while np.sum(samples_per_segment_stratum[segment_idx]) < total_num_samples:
            samples_per_segment_stratum[segment_idx][strata_idx] += 1
            strata_idx = (strata_idx + 1) % num_strata

    return np.array(samples_per_segment_stratum)


def interleave(strata_gt_streams, strata_proxy_streams, strata_predicate_streams, segment_idx, samples_per_segment_stratum, num_strata):
    print(f"interleaving segment: {segment_idx}")

    gt_stream, proxy_stream, predicate_stream = [], [], []
    for sample in range(int(np.sum(samples_per_segment_stratum[segment_idx]))):
        strata_to_sample = np.random.choice(
            list(range(num_strata)),
            p=samples_per_segment_stratum[segment_idx]/np.sum(samples_per_segment_stratum[segment_idx])
        )
        while len(strata_gt_streams[strata_to_sample]) == 0:
            strata_to_sample = (strata_to_sample + 1) % num_strata

        gt_sample = strata_gt_streams[strata_to_sample].pop()
        proxy_sample = strata_proxy_streams[strata_to_sample].pop()
        predicate_sample = strata_predicate_streams[strata_to_sample].pop()

        gt_stream.append(gt_sample)
        proxy_stream.append(proxy_sample)
        predicate_stream.append(predicate_sample)
    
    return gt_stream, proxy_stream, predicate_stream


def create_stream(num_shifts, vec_idx, num_strata):
    # seed random generator
    np.random.seed(12345 * num_shifts + vec_idx)

    # sample the locations in the stream where the shifts will occur
    shift_indices = sorted(list(np.random.choice(np.arange(MAX_FRAMES), size=num_shifts, replace=False)))

    # compute samples per stream
    samples_per_segment_stratum = compute_samples_per_segment_stratum(shift_indices, num_strata)
    
    # compute initial parameters;
    # - strata means are chosen uniformly at random from [0,3], [3,6], and [6,9]
    # - strata stds. are chosen uniformly at random from [0,3]
    # - strata predicate positivity rates are chosen uniformly at random from [0,1]
    strata_means = [3*(strata_idx + np.random.uniform()) for strata_idx in range(num_strata)]
    strata_stds = [3*np.random.uniform() for _ in range(num_strata)]
    strata_ps = [np.random.uniform() for _ in range(num_strata)]

    # compute groundtruth
    full_gt_stream, full_proxy_stream, full_predicate_stream = [], [], []
    for segment_idx in range(len(shift_indices) + 1):
        strata_gt_streams, strata_proxy_streams, strata_predicate_streams = [], [], []
        for strata_idx in range(num_strata):
            # compute groundtruth values
            strata_gt_stream = np.random.normal(
                loc=strata_means[strata_idx],
                scale=strata_stds[strata_idx],
                size=(1, int(samples_per_segment_stratum[segment_idx][strata_idx])),
            )
            strata_gt_streams.append(list(strata_gt_stream[0]))

            # copy groundtruth --> proxy values; we will mix these with random noise after normalizing
            strata_proxy_stream = strata_gt_stream
            strata_proxy_streams.append(list(strata_proxy_stream[0]))

            # compute predicate values
            strata_predicate_stream = np.random.binomial(1, strata_ps[strata_idx], size=(1, int(samples_per_segment_stratum[segment_idx][strata_idx])))
            strata_predicate_streams.append(list(strata_predicate_stream[0]))

        # interleave samples from each strata
        segment_gt_stream, segment_proxy_stream, segment_predicate_stream = interleave(strata_gt_streams, strata_proxy_streams, strata_predicate_streams, segment_idx, samples_per_segment_stratum, num_strata)

        # add stream of samples for segment to full dataset stream
        full_gt_stream.extend(segment_gt_stream)
        full_proxy_stream.extend(segment_proxy_stream)
        full_predicate_stream.extend(segment_predicate_stream)
        
        # sample new parameters
        strata_means = [3*(strata_idx + np.random.uniform()) for strata_idx in range(num_strata)]
        strata_stds = [np.random.uniform() for _ in range(num_strata)]
        strata_ps = [np.random.uniform() for _ in range(num_strata)]

    # normalize proxy values
    full_proxy_stream = full_proxy_stream - np.min(full_proxy_stream)
    full_proxy_stream = full_proxy_stream / np.max(full_proxy_stream)

    # add random noise to proxy values
    full_proxy_stream = BETA * full_proxy_stream + (1.0 - BETA) * np.random.uniform(0, 1, len(full_proxy_stream))

    return full_gt_stream, full_proxy_stream, full_predicate_stream


if __name__ == "__main__":
    # construct synthetic datasets
    for num_shifts in range(1, 6):
        for vec_idx in range(20):
            print(f"----- num_shifts: {num_shifts} -- vec: {vec_idx} ------")
            gt_stream, proxy_stream, predicate_stream = create_stream(num_shifts, vec_idx, NUM_STRATA)

            pd.DataFrame({"obj_count": gt_stream, "predicate": pred_stream}).to_csv(f"datasets/final-synthetic/true_num_shifts_{num_shifts}_vec_{vec_idx}.csv", index=False, header=False)
            pd.DataFrame({"proxy": proxy_stream}).to_csv(f"datasets/final-synthetic/prob_num_shifts_{num_shifts}_vec_{vec_idx}.csv", index=False, header=False)
