from os import error
import torch 
import torch.nn as nn
from network import Encoder
import numpy as np 
from evaluate_dataloader import LoadDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from siren_pytorch import SirenNet

model_dir = "checkpoints/10_2048_sigmoid/model_100.pth"
test_data_dir = "datasets/wind/testing.npy"
target_data_dir = "datasets/wind/gt_wind.npy"
num_layers = 10
num_hidden = 2048
activation = "sigmoid"
if activation == "identity":
    activation_func = nn.Identity()
elif activation == "ReLU":
    activation_func = nn.ReLU()
elif activation == "sigmoid":
    activation_func = nn.Sigmoid()
elif activation == "Tanh":
    activation_func = nn.Tanh()
else:
    print("Activation Function Error")

## Load Testing Data 
batch_size = 200
test_data = np.load(test_data_dir)
print(test_data.shape)
dataset = LoadDataSet(test_data[:, 0:8])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

# m_min = np.min(test_data[:, 0:4])
# m_max = np.max(test_data[:, 0:4])
# print("mean", m_min, m_max)
# w_min = np.min(test_data[:, 4:8])
# w_max = np.max(test_data[:, 4:8])
# print("width", w_min, w_max)

## Load Model 
# model = Encoder()
model = SirenNet(
    dim_in = 8,                        # input dimension, ex. 2d coor
    dim_hidden = num_hidden,                  # hidden dimension
    dim_out = 1,                       # output dimension, ex. rgb value
    num_layers = num_layers,                    # number of layers
    final_activation = activation_func,   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    model.load_state_dict(torch.load(model_dir, map_location=device))
else:
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_dir))
model.to(device)
print("Model Loaded!")

pred_results = empty_array = np.array([])
test_bar = tqdm(enumerate(dataloader))
model.eval()
for i, data in test_bar:
    mv = data.to(device)
    pred = model(mv)
    pred_cpu = pred.detach().cpu().numpy()
    # print("pred_cpu", pred_cpu.shape)
    pred_results = np.append(pred_results, pred_cpu)


# gt = test_data[:, 8]
# errors = np.absolute(gt - pred_results)
# error_1 = []
# error_2 = []
# for i, data in enumerate(test_data):
#     mean = data[0:4]
#     width = data[4:8]
#     m_min = np.min(mean)
#     m_max = np.max(mean)
#     w_min = np.min(width)
#     w_max = np.max(width)
#     if (m_min >=0.2 and m_min <= 0.8) and (w_min >=0.02 and w_max <= 0.2):
#         error_1.append(errors[i])
#     else:
#         error_2.append(errors[i])
#         if errors[i] > 0.1:
#             print("width", width)
#             print(w_min, w_max, errors[i])

# error_1 = np.array(error_1)
# error_2 = np.array(error_2)

# print("Minimum error: ", np.min(error_1))
# print("Maximum, error: ", np.max(error_1))
# print("Median error: ", np.median(error_1))
# print("Minimum error: ", np.min(error_2))
# print("Maximum, error: ", np.max(error_2))
# print("Median error: ", np.median(error_2))


# # print("Minimum error: ", np.min(errors))
# # print("Maximum, error: ", np.max(errors))
# # print("Median error: ", np.median(errors))
# sorted_errors = np.sort(errors)[0:errors.shape[0]-5]

# fig, ax = plt.subplots()
# # Create a plot
# g1 = ax.violinplot([sorted_errors], [1], showmeans=True, showmedians=True)
# # Add title
# ax.set_title('Violin Plot of 500 Testing Samples')
# ax.get_xaxis().set_visible(False)
# plt.show()

#(67, 67)
# (60, 508)
pred_results = np.reshape(pred_results, (67, 67))
target = np.load(target_data_dir)
diff = np.abs(target - pred_results)
# print("min max", np.min(diff), np.max(diff), np.mean(diff), np.median(diff))
count = 0
error_1 = [] 
error_2 = []

for i, data in enumerate(test_data):
    mean = data[0:4]
    std = data[4:8]
    if np.min(mean) >=0.06 and np.max(mean) <= 0.94 and np.min(std) >= 0.01 and np.max(std) <= 0.06:
        y = i // 67
        x = i % 67
        error_1.append(diff[x, y])
        if diff[x, y] > 0.5:
            print(target[x, y], pred_results[x, y])
    else:
        y = i // 67
        x = i % 67
        error_2.append(diff[x, y])
error_1 = np.array(error_1)
error_2 = np.array(error_2)
print("error 1:", error_1.shape, np.min(error_1), np.max(error_1), np.mean(error_1))
print("error 2:", error_2.shape, np.min(error_2), np.max(error_2), np.mean(error_2))

for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        error = diff[i, j]
        if error > 0.1:
            # print(test_data[j * 67 + i], error)
            count+=1
print("count: ", count, count / (diff.shape[0] * diff.shape[1]) * 100)

# fig = plt.figure(figsize=(50, 6))
axs = (plt.figure(figsize=(10, 30), constrained_layout=True).subplots(3, 1, sharex=True, sharey=False))
axs[0].imshow(target, vmin=0, vmax =1)
# plt.colorbar(img, shrink=0.5)

# plt.show()

# plt.tight_layout()
axs[1].imshow(pred_results, vmin = 0, vmax = 1)
# plt.colorbar(img)
# plt.show()

# plt.tight_layout()
axs[2].imshow(diff, cmap="gray", vmin=0, vmax =1.0)
# plt.colorbar(img)
# plt.tight_layout()
plt.show()