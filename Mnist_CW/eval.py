import numpy as np
import torch
from tqdm import tqdm

from CarliniL2 import CWL2Attack
from test_attack import generate_data, predict
from train import MNISTDataset


def evalu(model, inputs, targets, ground_truth, device):
    model.eval()
    total = len(ground_truth)
    valid_mapping = np.ones((total,2))
    attacker = CWL2Attack(model, device, False, 1, 0, 100, 0.01)
    success = 0
    idx = 0
    its = 100
    with torch.no_grad():
        for step in tqdm(range(len(ground_truth)), desc="Getting Ready", ncols=100):
            if predict(model,inputs[step].to(device)) != ground_truth[step]:
                valid_mapping[step][0] = 0
                valid_mapping[step][1] = 0
    for step in tqdm(range(its), desc="Evaluating", ncols=100):
        if valid_mapping[idx][0] == 1:
            if predict(model, attacker.attack(inputs[idx], ground_truth[idx], False)) != ground_truth[idx]:
                success+=1
                step+=1
                valid_mapping[idx][1] = 0
        idx += 1
    print(success * 100 / its,'%')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = MNISTDataset()
    model = torch.load('./models/mnist.pth')
    inputs, targets, ground_truth = generate_data(data, len(data.get_test_data()))
    model.eval()
    model.to(device)
    evalu(model, inputs, targets,ground_truth, device)

if __name__ == '__main__':
    main()


# c = 0.1
# Getting Ready: 100%|████████████████████████████████████████| 10000/10000 [00:03<00:00, 2944.43it/s]
# Evaluating: 100%|█████████████████████████████████████████████████| 100/100 [00:16<00:00,  6.08it/s]
# 9.0 %

# c = 1
# Getting Ready: 100%|████████████████████████████████████████| 10000/10000 [00:03<00:00, 2542.39it/s]
# Evaluating: 100%|█████████████████████████████████████████████████| 100/100 [00:17<00:00,  5.87it/s]
# 68.0 %

# c = 10
# Getting Ready: 100%|████████████████████████████████████████| 10000/10000 [00:04<00:00, 2457.20it/s]
# Evaluating: 100%|█████████████████████████████████████████████████| 100/100 [00:17<00:00,  5.67it/s]
# 91.0 %
