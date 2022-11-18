import numpy as np
import time
import torch 
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader

# from fourier_feature_transform import GaussianFourierFeatureTransform      

class LoadDataSet(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data = []
        data = np.load(data_dir)
        num_samples = int(data.shape[0] * 1)
        # indices = np.random.choice(data.shape[0], num_samples, replace=False)
        p = data[0:num_samples, 15]
        means = data[0:num_samples, 0:4] 
        covs = data[0:num_samples, 4:14]
        iso = data[0:num_samples, 14]
            
        ## normalize to 0 to 1
        p_min = np.min(p)
        p_max = np.max(p)
        # mv_min = np.min(mv_list)
        # mv_max = np.max(mv_list)
        # print("mv", mv_min, mv_max)
        print("p", p_min, p_max)
        m_min = np.min(means)
        m_max = np.max(means)
        v_min = np.min(covs)
        v_max = np.max(covs)
        print("m", m_min, m_max)
        print("v", v_min, v_max)

        ## scale to [t_min, t_max]
        t_min = 0
        t_max = 1
        for i, m in enumerate(means):

            m_normalized = (means[i] - m_min) / (m_max - m_min) * (t_max - t_min) + t_min
            v_normalized = (covs[i] - v_min) / (v_max - v_min) * (t_max - t_min) + t_min
            p_normalized = p[i]
            iso_temp = (iso[i] - 0.1) / (0.9 - 0.1) * (t_max - t_min) + t_min
            self.data.append({
                "m": torch.FloatTensor(m_normalized),
                "v": torch.FloatTensor(v_normalized),
                "iso": torch.FloatTensor([iso_temp]),
                "p": torch.FloatTensor([p_normalized])
            })
    def __len__(self):
        print("data size:", len(self.data))
        return len(self.data)
    
    def __getitem__(self, index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        m = data["m"]
        v = data["v"]
        iso = data["iso"]
        p = data["p"]
        return m, v, iso, p

if __name__ == "__main__":
    data_dir = "datasets/npy/trainingData10kSamples.npy"
    dataset = LoadDataSet(data_dir)
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, drop_last=True)
    m, v, p = next(iter(train_dataloader))
    print(m)
    print(v)
    print(p)
