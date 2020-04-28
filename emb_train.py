from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from emb_modeling import ContextStaticModel
from emb_dataset import Mydataset
import torch
from torch.nn import functional as F
import json
import time
import numpy as np
from emb_config import model_config,train_config,data_config,eval_config


def get_data(data_path, vocabulary_path, stop_words_path):
    data = torch.load(data_path)
    with open(stop_words_path, encoding='utf8') as f:
        stop_words = []
        for i in f:
            stop_words.append(i.strip())
    with open(vocabulary_path, encoding='utf8') as f:
        vocabulary_dic = json.load(f)
    vocabulary = []
    for k, v in vocabulary_dic.items():
        vocabulary.append(k)
    src = data['train']['src']
    accusation = data['train']['accusation']
    return src, accusation, stop_words, vocabulary_dic, vocabulary


def loss_function(x_bow, out, mu, var):
    # loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    # BCE_bow = loss_fn(x_bow, out)
    BCE = -torch.sum(x_bow * F.log_softmax(out,dim=-1), dim=-1)
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

    return  torch.sum(BCE), torch.sum(KLD)


def train(dataset, model, train_config, eval_config, model_config):
    dataLoader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.zero_grad()
    train_iterator = trange(1, int(train_config['epochs']), desc="TRAIN Epoch")
    for ep in train_iterator:
        for step, batch in enumerate(dataLoader):
            x_bow, x_seq, mask, _ = batch
            x_bow = x_bow.to(device)
            x_seq = x_seq.to(device)
            mask = mask.to(device)
            model.train()
            out, mu, var, z, x = model(x_bow, x_seq, mask, device)
            rc_loss, kl_loss = loss_function(x_bow, out, mu, var)
            total_loss = (rc_loss + kl_loss)/1000
            total_loss.backward()
            print('**EPOCH:{}**STEP:{}**'.format(ep, step))
            print('** rc_loss:{}'.format(rc_loss.item()))
            print('** kl_loss:{}'.format(kl_loss.item()))
            print('** total_loss:{}'.format(total_loss.item()))
            print('*********************')
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['clip_grad'])
            optimizer.step()
            model.zero_grad()
        if ep % train_config['save_step'] == 0:
            torch.save(model.state_dict(), 'checkpoint/params' + str(ep) + '.bin')
            model_config['is_traing'] = False
            eval_model = ContextStaticModel(**model_config)
            eval_model.load_state_dict(torch.load('checkpoint/params' + str(ep) + '.bin', map_location='cpu'))
            evaluate(eval_model, dataset, eval_config, ep)


def evaluate(model, dataset, eval_config, epoch):
    with open('log/result_log.txt', 'a') as f:
        f.write('***********{}***********{}\n'.format(epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    eval_dataloader = DataLoader(dataset, batch_size=eval_config['batch_size'], shuffle=False)
    model.to('cpu')
    eval_iterator = tqdm(eval_dataloader, desc="Evaluate Iteration")
    all_out_list = []
    accusation_list = []
    for step, batch in enumerate(eval_iterator):
        x_bow, x_seq, mask, accusation = batch
        x_bow = x_bow.to('cpu')
        x_seq = x_seq.to('cpu')
        mask = mask.to('cpu')
        model.eval()
        mu = model(x_bow, x_seq, mask, 'cpu')[1].detach().numpy()  # out, mu, var, z, x
        all_out_list.extend(mu)
        accusation_list.extend(accusation)

    # test case accuracy
    set_accusation = set(accusation_list)
    print(set_accusation)
    all_out_list = torch.from_numpy(np.array(all_out_list))
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    case_accuracy = []
    for case in set_accusation:
        num_case = 0
        sum_all = 0
        for index, i in enumerate(accusation_list[:]):
            if case in i:
                all_num_case = accusation_list.count(case)
                num_case += 1
                test_cos = cos(all_out_list[index], all_out_list)
                test_sort = torch.sort(test_cos, descending=True)
                suma = 0
                #print(test_sort[0])
                for j in test_sort[1][1:all_num_case]:
                    if accusation_list[j.item()] in [accusation_list[test_sort[1][0].item()]]:
                        suma += 1
                sum_all += suma
                # print('案件编号{}，准确率{}'.format(index,suma/20))
        case_accuracy.append(sum_all / num_case / all_num_case)
        print('{},案件个数{}，正确率{}'.format(case, num_case, sum_all / num_case / all_num_case))
        with open('log/result_log.txt', 'a') as f:
            f.write('{},案件个数{}，正确率{}\n'.format(case, num_case, sum_all / num_case / all_num_case))
    with open('log/result_log.txt', 'a') as f:
        f.write('平均值为：{}\n'.format(sum(case_accuracy)/10))


if __name__ == '__main__':

    src, accusation, stop_words, vocabulary_dic, vocabulary = get_data(**data_config)
    print("vocabulary")
    dataset = Mydataset(src, accusation, model_config['seq_len'], vocabulary_dic, stop_word=stop_words,
                         vocabulary=vocabulary)
    if model_config['is_traing'] is not True:
        raise ValueError("Train flag must true before yor train model")
    model = ContextStaticModel(**model_config)
    train(dataset, model, train_config, eval_config, model_config)