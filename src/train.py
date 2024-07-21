import random
import numpy as np
import os
import torch as torch
from src.load_data import load_EOD_data
from src.evaluator import evaluate
from src.model import get_loss, StockMixer
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

data_path = '../dataset'
market_name = 'NASDAQ'
stock_num = 909
lookback_length = 16
valid_index = 756
test_index = 1008
fea_num = 5
market_num = 20
steps = 1
learning_rate = 0.001
scale_factor = 3
activation = 'GELU'

dataset_path = 'dataset/' + market_name
if market_name == "SP500":
    data = np.load('dataset/SP500/SP500.npy')
    data = data[:, 915:, :]
    price_data = data[:, :, -1]
    mask_data = np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   data[ticket][row - steps][-1]
else:
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)


def validate(model, start_index, end_index, alpha, beta):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        classification_loss = 0.
        all_pred_probs = []
        all_gt_labels = []
        res = pd.DataFrame()
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_classification_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                                             stock_num, alpha, beta)
            resi = pd.DataFrame({'prediction': cur_rr.cpu()[:, 0], 'ground_truth': gt_batch.cpu()[:, 0]})
            resi['id'] = cur_offset
            res = pd.concat([res, resi], ignore_index=True)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            classification_loss += cur_classification_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
            
            # Collect predictions and ground truth for classification metrics
            pred_probs = torch.sigmoid(cur_rr * 100).cpu().numpy()
            gt_labels = (gt_batch > 0).float().cpu().numpy()
            all_pred_probs.extend(pred_probs)
            all_gt_labels.extend(gt_labels)

        # Calculate classification metrics
        all_pred_probs = np.array(all_pred_probs).flatten()
        all_gt_labels = np.array(all_gt_labels).flatten()
        all_pred_labels = (all_pred_probs > 0.5).astype(int)
        
        accuracy = accuracy_score(all_gt_labels, all_pred_labels)
        precision = precision_score(all_gt_labels, all_pred_labels, zero_division=0)
        recall = recall_score(all_gt_labels, all_pred_labels, zero_division=0)
        
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        classification_loss = classification_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, classification_loss, cur_valid_perf, res, accuracy, precision, recall


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))


def train(model, epochs=100, alpha=5, beta=1e-3):
    trade_dates = mask_data.shape[1]
    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)
    final_val_res = None
    final_test_res = None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print("epoch{}##########################################################".format(epoch + 1))
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        tra_classification_loss = 0.0
        for j in tqdm(range(valid_index - lookback_length - steps + 1)):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_classif_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                                  stock_num, alpha, beta)
            cur_loss = cur_loss
            cur_loss.backward()
            optimizer.step()
            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()
            tra_classification_loss += cur_classif_loss.item()
        tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
        tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
        tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
        tra_classification_loss = tra_classification_loss / (valid_index - lookback_length - steps + 1)
        print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e} + beta*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss, tra_classification_loss))
    
        val_loss, val_reg_loss, val_rank_loss, val_classification_loss, val_perf, val_res, val_acc, val_prec, val_rec = validate(model, valid_index, test_index, alpha=alpha, beta=beta)
        print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e} + beta*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss, val_classification_loss))
        print(f'Valid classification metrics: Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}')
    
        test_loss, test_reg_loss, test_rank_loss, test_classification_loss, test_perf, test_res, test_acc, test_prec, test_rec = validate(model, test_index, trade_dates, alpha=alpha, beta=beta)
        print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e} + beta*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss, test_classification_loss))
        print(f'Test classification metrics: Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}')
    
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf
            final_val_res = val_res
            final_test_res = test_res
    
        print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                         val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
        print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                                                           test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')
    return val_res, test_res
