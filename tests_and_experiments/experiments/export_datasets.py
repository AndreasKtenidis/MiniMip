from tests_and_experiments.datasets.blob import BlobDataset
import numpy as np
from sklearn.datasets import make_classification

for client_id in range(0,3):
    for sample_size in [5882352,58823520,588235200]:
        blob_dataset = BlobDataset(n_samples=sample_size, centers=3).get_local_dataset(client_id, 3).values
        np.savetxt(f'/data/dataset/csvs/blob_dataset{sample_size}_{client_id}.npy', blob_dataset,delimiter=',')
        # --------------------------------
        x_logistic, y_logistic = make_classification(n_samples=sample_size, n_features=20, n_informative=15, n_redundant=5, n_classes=2,
                                   random_state=42 * client_id)
        # --------------------------------
        merged = np.hstack([x_logistic, y_logistic.reshape(-1, 1)
])
        np.savetxt(f'/data/dataset/csvs/x_y_logistic_{sample_size}_{client_id}.npy', merged,delimiter=',')
        # --------------------------------
        rand_data, _ = make_classification(n_samples=sample_size, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                           random_state=42 * client_id)
        np.savetxt(f'/data/dataset/csvs/x_y_corr_{sample_size}_{client_id}.npy', rand_data,delimiter=',')