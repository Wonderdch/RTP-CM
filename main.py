import os
import pickle
import random
import time

import numpy as np
import torch

from RTP_CM import RTP_CM
from mask_strategy import Mask
from parameter_parser import parse
from result.data_reader import print_output_to_file, calculate_average, clear_log_meta_model


def train_RTP_CM(train_set, test_set, h_params, vocab_size, device, run_name):
    model_path = f"./result/{run_name}_model"
    log_path = f"./result/{run_name}_log"
    meta_path = f"./result/{run_name}_meta"

    print("parameters:", h_params)

    if os.path.isfile(f'./results/{run_name}_model'):
        try:
            os.remove(f"./results/{run_name}_meta")
            os.remove(f"./results/{run_name}_model")
            os.remove(f"./results/{run_name}_log")
        except OSError:
            pass
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()

    # Construct model
    model = RTP_CM(
        vocab_size=vocab_size,
        area_code_embed_size=h_params['geohash_embed_size'],
        area_proportion=h_params['area_proportion'],
        feature_embed_size=h_params['embed_size'],
        transformer_layers=h_params['transformer_layers'],
        transformer_heads=h_params['transformer_heads'],
        forward_expansion=h_params['expansion'],
        dropout_proportion=h_params['dropout'],
        back_step=h_params['back_step'],
        mask_strategy=h_params['mask_strategy'],
        mask_proportion=h_params['mask_proportion'],
        device=h_params['device'])

    model = model.to(device)

    params = list(model.parameters())

    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}

    for i in range(0, h_params['epochs']):
        begin_time = time.time()
        total_loss = 0.
        for sample in train_set:
            sample_to_device = []
            # [(seq1)[((features)[poi_seq],[cat_seq],[user_seq],[hour_seq],[day_seq]),([area_codes 0~5])],[(seq2)],...]
            for seq in sample:
                features = torch.tensor(seq[:5]).to(device)
                area_codes = torch.tensor(seq[5:10]).to(device)
                sample_to_device.append((features, area_codes))

            loss, _ = model(sample_to_device)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        recall, ndcg, map = test_RTP_CM(test_set, model)
        recalls[i] = recall
        ndcgs[i] = ndcg
        maps[i] = map

        # Record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[i] = avg_loss
        print(f"epoch: {i}; average loss: {avg_loss}, time taken: {int(time.time() - begin_time)}s")
        # Save model
        torch.save(model.state_dict(), model_path)
        # Save last epoch
        meta_file = open(meta_path, 'wb')
        pickle.dump(i, meta_file)
        meta_file.close()

        log_file = open(log_path, 'wb')
        pickle.dump(loss_dict, log_file)
        pickle.dump(recalls, log_file)
        pickle.dump(ndcgs, log_file)
        pickle.dump(maps, log_file)
        log_file.close()

    print("============================")


def test_RTP_CM(test_set, rec_model, ks=[1, 5, 10]):
    def calc_hit_rate(labels, preds, k):
        hit = []
        i = 0
        for label in labels:
            predictions = preds[i, :k]
            if label in predictions:
                hit.append(1.0)
            else:
                hit.append(0.0)
            i += 1
        hit_rate = np.mean(hit)
        return hit_rate

    def calc_recall(labels, preds, k):
        equal_count = torch.sum(labels == preds[:, :k], dim=1)
        sum_equal_count = torch.sum(equal_count)
        labels_shape0 = labels.shape[0]
        recall = sum_equal_count / labels_shape0
        return recall

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    for sample in test_set:
        sample_to_device = []
        # [(seq1)[((features)[poi_seq],[cat_seq],[user_seq],[hour_seq],[day_seq]),([area_codes 0~5])],[(seq2)],...]
        for seq in sample:
            features = torch.tensor(seq[:5]).to(device)
            area_codes = torch.tensor(seq[5:10]).to(device)
            sample_to_device.append((features, area_codes))

        pred, label = rec_model.predict(sample_to_device)
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        hit_rate = calc_hit_rate(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tHR @{k} : {hit_rate},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")

    return recalls, NDCGs, MAPs


if __name__ == '__main__':
    args = parse()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Get parameters
    parameters = {
        'device': args.device,
        'mask_strategy': Mask(args.mask_strategy),
        'mask_proportion': args.mask_proportion,
        'area_proportion': args.area_proportion,
        'embed_size': args.embed_size,
        'transformer_layers': args.transformer_layers,
        'transformer_heads': args.transformer_heads,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'lr': args.lr,
        'expansion': 4}

    # Adjust specific parameters for each city
    if args.dataset == 'PHO':
        parameters['geohash_embed_size'] = {"0": 6, "1": 60, "2": 508, "3": 1075, "4": 1367}
        parameters['back_step'] = 1
    elif args.dataset == 'NYC':
        parameters['geohash_embed_size'] = {"0": 8, "1": 67, "2": 1042, "3": 5310, "4": 12927}
        # Keep back_step for a fair comparison with CFPRec
        parameters['back_step'] = 2
    elif args.dataset == 'SIN':
        parameters['geohash_embed_size'] = {"0": 2, "1": 24, "2": 303, "3": 2615, "4": 6273}
        parameters['back_step'] = 2
    else:
        raise NotImplementedError()

    # Read training data
    file = open(f"./processed_data/{args.dataset}_train", 'rb')
    train_set = pickle.load(file)
    file = open(f"./processed_data/{args.dataset}_valid", 'rb')
    valid_set = pickle.load(file)

    # Read meta data
    file = open(f"./processed_data/{args.dataset}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {
        "POI": torch.tensor(len(meta["POI"])).to(device),
        "cat": torch.tensor(len(meta["cat"])).to(device),
        "user": torch.tensor(len(meta["user"])).to(device),
        "hour": torch.tensor(len(meta["hour"])).to(device),  # 24
        "day": torch.tensor(len(meta["day"])).to(device)}  # 2

    print(f'Current GPU {args.device}')
    for run_num in range(1, 1 + args.run_times):
        run_name = f'{args.name} {run_num}'
        print(run_name)

        train_RTP_CM(train_set, valid_set, parameters, vocab_size, device, run_name=run_name)
        print_output_to_file(args.name, run_num, args.epochs)

        t = random.randint(1, 9)
        print(f"sleep {t} seconds")
        time.sleep(t)

        clear_log_meta_model(args.name, run_num)
    calculate_average(args.name, args.run_times)
