import torch
from model import Model
from data_helper import DataHelper
import numpy as np
import tqdm
import sys, random
import argparse
import time, datetime
import os
from pmi import cal_PMI
from sklearn.metrics import classification_report

NUM_ITER_EVAL = 100
EARLY_STOP_EPOCH = 25
learning_rate = 0.01


def edges_mapping(vocab_len, content, ngram):
    count = 1
    mapping = np.zeros(shape=(vocab_len, vocab_len), dtype=np.int32)
    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i - ngram), min(len(doc), i + ngram + 1)):
                dst = doc[dst_id]

                if mapping[src, dst] == 0:
                    mapping[src, dst] = count
                    count += 1

    for word in range(vocab_len):
        mapping[word, word] = count
        count += 1

    return count, mapping


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset):
    data_helper = DataHelper(dataset, mode='dev')

    total_pred = 0
    correct = 0
    iter = 0
    for content, label, dep, _ in data_helper.batch_iter(batch_size=32, num_epoch=1):  # 原batch_size=64, num_epoch=1
        iter += 1
        model.eval()

        logits = model(content, dep)
        pred = torch.argmax(logits, dim=1)

        correct_pred = torch.sum(pred == label)

        correct += correct_pred
        total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    # print(torch.div(correct, total_pred))
    return torch.div(correct, total_pred)

def correct_print(label, pred):
    c = []
    for i in range(len(label)):
        if label[i] == pred[i]:
            c.append(label[i])
    t_class = np.unique(label)
    c_0 = 0
    c_1 = 0
    for j in c:
        if j==t_class[0]:
            c_0 += 1
        else:
            c_1 += 1
    print('correct', t_class[0], ':', c_0)
    print('correct', t_class[1], ':', c_1)


def test(model_name, dataset):
    model = torch.load(os.path.join('.', model_name + '.pkl'))

    data_helper = DataHelper(dataset, mode='test')

    total_pred = 0
    correct = 0
    iter = 0
    total_preds = []
    total_label = []
    for content, label, dep, _ in data_helper.batch_iter(batch_size=32, num_epoch=1):  # 原batch_size=64, num_epoch=1
        iter += 1
        model.eval()

        logits = model(content, dep)
        pred = torch.argmax(logits, dim=1)

        correct_pred = torch.sum(pred == label)

        correct += correct_pred
        total_pred += len(content)

        total_preds = np.concatenate([total_preds,pred.to('cpu')])
        total_label = np.concatenate([total_label, label.to('cpu')])

    total_pred = float(total_pred)
    correct = correct.float()
    # print(torch.div(correct, total_pred))
    correct_print(total_label, total_preds)
    metrices = classification_report(y_true = total_label, y_pred = total_preds, digits = 6)
    return torch.div(correct, total_pred).to('cpu'), metrices


def train(ngram, name, bar, drop_out, dataset, is_cuda=True, edges=True):
    print('load data helper.')
    data_helper = DataHelper(dataset, mode='train')
    if os.path.exists(os.path.join('.', name + '.pkl')) and name != 'temp_model':
        print('load model from file.')
        model = torch.load(os.path.join('.', name + '.pkl'))
    else:
        print('new model.')
        if name == 'temp_model':
            name = 'temp_model_%s' % dataset
        # edges_num, edges_matrix = edges_mapping(len(data_helper.vocab), data_helper.content, ngram)
        edges_weights, edges_mappings, count = cal_PMI(dataset=dataset)

        model = Model(class_num=len(data_helper.labels_str), hidden_size_node=300,
                      vocab=data_helper.vocab, n_gram=ngram, drop_out=drop_out, edges_matrix=edges_mappings,
                      edges_num=count,
                      trainable_edges=edges, pmi=edges_weights, cuda=is_cuda)

    print(model)
    if is_cuda:
        print('cuda')
        model.cuda()
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-8)  # 修改L2正则化，原来为1e-6, 增加learning——rate参数 0.001

    # optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8,alpha=0.9)

    # 根据网址调节的可自变化的学习率 https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)  # 指数衰减，每次epoch过后学习率乘0.9，还有其它方式，可以都试一下

    iter = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best_acc = 0.0
    best_tacc = 0.0
    last_best_epoch = 0
    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total = 0

    num_epoch = 4
    batch_size = 96
    for i in range(num_epoch):  # 训练参数，原batch_size=32, num_epoch=200
        total_content, total_label, total_dep = data_helper.get_review()
        num_per_epoch = int(len(total_content) / batch_size)
        for batch_id in range(num_per_epoch):
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, len(total_content))

            content = total_content[start:end]
            dep = total_dep[start:end]
            temp_label = total_label[start:end]
            label = torch.tensor(temp_label).cuda()
            improved = ''
            model.train()
            # batch_iter是根据batchsize将数据集划分，每次都会返回一部分数据进行训练
            logits = model(content, dep)
            torch.cuda.empty_cache()  # 后期添加
            loss = loss_func(logits, label)
            # 训练预测的标签
            pred = torch.argmax(logits, dim=1)

            correct = torch.sum(pred == label)

            total_correct += correct
            total += len(label)

            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss)
            iter += 1
            if bar:
                pbar.update()
            if iter % NUM_ITER_EVAL == 0:
                if bar:
                    pbar.close()

                val_acc = dev(model, dataset=dataset)
                if val_acc > best_acc:
                    best_acc = val_acc
                    last_best_epoch = i
                    improved = '*'

                    torch.save(model, name + '.pkl')


                if i - last_best_epoch >= EARLY_STOP_EPOCH:
                    return name
                msg = 'Epoch: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Train Acc: {6:>7.2%}' \
                      + 'Val Acc: {2:>7.2%}, Time: {3}{4}' \
                    # + ' Time: {5} {6}'

                print(msg.format(i, iter, val_acc, get_time_dif(start_time), improved,total_loss / NUM_ITER_EVAL,
                                 float(total_correct) / float(total)))

                total_loss = 0.0
                total_correct = 0
                total = 0
                if bar:
                    pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
        if i < 2:  # 更新两次到0.001
            scheduler.step()  # 更新学习率


    return name


def word_eval():
    print('load model from file.')
    data_helper = DataHelper('r8')
    edges_num, edges_matrix = edges_mapping(len(data_helper.vocab), data_helper.content, 1)
    model = torch.load(os.path.join('word_eval_1.pkl'))

    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()

    core_word = 'billion'
    core_index = data_helper.vocab.index(core_word)

    results = {}
    for i in range(len(data_helper.vocab)):
        word = data_helper.vocab[i]
        n_word = edges_matrix[i, core_index]
        # n_word = edges_matrix[i, i]
        if n_word != 0:
            results[word] = edges_weights[n_word][0]

    sort_results = sorted(results.items(), key=lambda d: d[1])

    print(sort_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', required=False, type=int, default=4, help='ngram number')
    parser.add_argument('--name', required=False, type=str, default='temp_model', help='project name')
    parser.add_argument('--bar', required=False, type=int, default=0, help='show bar')  # 显示进度条
    parser.add_argument('--dropout', required=False, type=float, default=0.5, help='dropout rate')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--edges', required=False, type=int, default=1, help='trainable edges')
    parser.add_argument('--rand', required=False, type=int, default=7, help='rand_seed')

    args = parser.parse_args()

    print('ngram: %d' % args.ngram)
    print('project_name: %s' % args.name)
    print('dataset: %s' % args.dataset)
    print('trainable_edges: %s' % args.edges)
    # #
    SEED = args.rand
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.bar == 1:
        bar = True
    else:
        bar = False

    if args.edges == 1:
        edges = True
        print('trainable edges')
    else:
        edges = False

    model = train(args.ngram, args.name, bar, args.dropout, dataset=args.dataset, is_cuda=True, edges=edges)
    acc, met = test(model, args.dataset)
    print('test acc: ', acc.numpy())
    print('metrices: ', met)
