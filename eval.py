import torch
from torch import nn
from model import *
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def evaluate(args, test_iter, pretrained_dir):
    # get net
    device = f'cuda:{args.cuda}' if args.cuda > 0 and torch.cuda.is_available() else 'cpu'
    net = CNN().to(device)
    net.load_state_dict(torch.load(pretrained_dir, map_location=device))
    
    # set loss function
    loss_func = nn.CrossEntropyLoss()
   
    with torch.no_grad():
        test_corr = 0
        print('Start evaluation on test set...')
        for _, (img, labels) in enumerate(tqdm(test_iter)): 
            net.eval() # test
            img, labels = img.to(device), labels.to(device)

            pred = net(img)
            loss = loss_func(pred, labels.long()).mean()

            _, pred = torch.max(pred.data, dim=1)
            batch_corr = pred.eq(labels.data).cpu().sum().item()
            test_corr += batch_corr

        test_acc = 100.0 * test_corr / args.batch_size / len(test_iter)

    print(f'test_acc = {test_acc:.4f}')