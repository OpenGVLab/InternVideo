from collections import Counter
import json
import random
import numpy as np
data = json.load(open("/path/to/to_sample"))
video_id = set([x["video"].split("/")[-1][:11] for x in data])
video_id_counter = Counter([x["video"].split("/")[-1][:11] for x in data])
sampling_weights = [1.0 / video_id_counter[x["video"].split("/")[-1][:11]] for x in data]
np.random.seed(42)
sampling_weights = np.array(sampling_weights)
sampling_weights = sampling_weights / sampling_weights.sum()
sampled_index = np.random.choice(len(data), 10647458, replace=False, p=sampling_weights)
data = [data[i] for i in sampled_index]
json.dump(data, open("/path/to/sampled", "w"))