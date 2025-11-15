
import os
import random
import time

import torch
import numpy as np

# Placeholder for model import (replace with your Mymodel.py import)
from models.model import Model  

# Import from local util.py
from util import HypergraphEHRDataset, format_time, MultiStepLRScheduler,load_adj, load_icd_adj

# Placeholder for metrics (adapt from Chet-master)
from metrics import evaluate_codes, evaluate_hf  

# Placeholder for historical_hot function
def historical_hot(code_x, code_num, lens):
    # TODO: Implement or adapt
    pass

if __name__ == '__main__':
    seed = 6669  #
    dataset = 'mimic3'  # Based on data path
    task = 'm'  

    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    device = torch.device('cuda:0')
    # Model hyperparameters (placeholders)
    code_size = 256
    graph_size = 32
    hidden_size = 256
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 200

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # #第二个level2，#第三个level2&3，第三个level 3

    # Data paths for the provided pkl files

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')
    code_adj = load_adj(dataset_path, device=device)
    icd_adj=load_icd_adj(dataset_path, device=device)
    code_num = len(code_adj)
    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    lambda_cl=0.005
    lambda_ont = 1e-3 
    print('loading train data ...')
    train_data = HypergraphEHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)

    print('loading valid data ...')
    valid_data = HypergraphEHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=True, device=device)

    print('loading test data ...')
    test_data = HypergraphEHRDataset(test_path, label=task, batch_size=batch_size, shuffle=True, device=device)

    # Placeholder for historical data
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    # Task configuration (similar to Chet-master)

    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    # Parameter saving path
    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    # Initialize model (replace with your Hypergraph model)
    model = Model(code_num=code_num, code_size=code_size,
                      adj=icd_adj, graph_size=graph_size, hidden_size=hidden_size,
                      t_attention_size=t_attention_size, t_output_size=t_output_size,
                      output_size=output_size, dropout_rate=dropout_rate,
                      activation=activation).to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    # Print parameter count
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # 初始化最好的性能跟踪变量
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, y = train_data[step]  # Unpack hypergraph batch
            output, loss_cl, lap_loss = model(code_x, visit_lens)
            output = output.squeeze(-1)
            loss = loss_fn(output, y) + lambda_cl * loss_cl + lambda_ont * lap_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
            # if (step % 80) == 0:
            #     print(f"\n      BCE={loss_fn(output, y).item():.4f}  CL={loss_cl.item():.4f}  λ={lambda_cl:.4f}  λ·CL={(lambda_cl*loss_cl).item():.4f}")
        train_data.on_epoch_end() 
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))


        # Evaluation (adapt to your metrics)
        valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)  # TODO: Adjust
        torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))

        # 更新和输出最好的F1分数
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epoch = epoch + 1
        print('    Current Best F1: %.4f (Epoch %d)' % (best_f1, best_epoch))
