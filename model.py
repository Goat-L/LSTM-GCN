import dgl 
import torch
import torch.nn.functional as F
import numpy as np
import word2vec
import tensorflow.keras as keras
from pmi import cal_PMI


# from dgl.data.tree import SST
# from dgl.data import SSTBatch


def gcn_msg(edge):
    return {'m': edge.src['h'], 'w': edge.data['w']}
    # 对图进行消息传播


def gcn_reduce(node):
    w = node.mailbox['w']

    new_hidden = torch.mul(w, node.mailbox['m'])

    new_hidden, _ = torch.max(new_hidden, 1)

    node_eta = torch.sigmoid(node.data['eta'])
    # node_eta = F.leaky_relu(node.data['eta'])

    # new_hidden = node_eta * node.data['h'] + (1 - node_eta) * new_hidden
    # print(new_hidden.shape)

    return {'h': new_hidden}


class Model(torch.nn.Module):
    def __init__(self,
                 class_num,
                 hidden_size_node,
                 vocab,
                 n_gram,
                 drop_out,
                 edges_num,
                 edges_matrix,
                 max_length=1000,
                 trainable_edges=True,
                 pmi=None,
                 cuda=True
                 ):
        super(Model, self).__init__()

        self.is_cuda = cuda
        self.vocab = vocab
        # 这个是词汇表，包含所有出现过的单词
        # print(len(vocab))
        self.seq_edge_w = torch.nn.Embedding(edges_num, 1)
        # 这个是边的权重，可学习，一开始全部设置为1，后期进行更新，维数为边的条数。
        print(edges_num)
        print(pmi.shape)

        self.node_hidden = torch.nn.Embedding(len(vocab), hidden_size_node)  # hidden_size_node设置为200

        # self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)  # 一开始边的权重设置为pmi，从pre——trained中获得

        self.edges_num = edges_num


        if trainable_edges:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=False)
            # self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.ones(edges_num, 1), freeze=False)
        else:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)

        # if trainable_edges:
        #     self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.ones(edges_num, 1), freeze=False)
        # else:
        #     self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)
        # 在这里对边的权重进行训练

        self.hidden_size_node = hidden_size_node

        self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec('glove.6B.300d.txt')))
        # 节点的权重是从这里进行发输入
        self.node_hidden.weight.requires_grad = True

        self.len_vocab = len(vocab)

        self.ngram = n_gram

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.max_length = max_length

        self.edges_matrix = edges_matrix

        self.dropout = torch.nn.Dropout(p=drop_out)

        lstmout_dim = 50

        self.lstmout_dim = lstmout_dim

        self.bilstm = torch.nn.LSTM(hidden_size_node, lstmout_dim,
                                    num_layers=1, batch_first=True, bidirectional=True)  # 200是词向量的维度， 100是处理后的词向量的维度
        # 由于BiLSTM会将前向和后向的输出concat起来，所以它最后的结果可能会乘2
        # 8 100 200--》 100,因为是双向50*2

        # 因为batch first，所以要封装在一起

        self.dropout = torch.nn.Dropout(p=drop_out)

        # self.gcn1 = dgl.nn.pytorch.conv.GraphConv(100, 100, norm='both', weight=True, bias=True)
        #
        # self.gcn2 = dgl.nn.pytorch.conv.GraphConv(100, 100, norm='both', weight=True, bias=True)
        #
        # self.gcn3 = dgl.nn.pytorch.conv.GraphConv(100, 100, norm='both', weight=True, bias=True)

        self.sage1 = dgl.nn.pytorch.conv.SAGEConv(in_feats=100, out_feats=100, aggregator_type='gcn', feat_drop=0.5)

        self.sage2 = dgl.nn.pytorch.conv.SAGEConv(in_feats=100, out_feats=100, aggregator_type='gcn', feat_drop=0.5)

        self.sage3 = dgl.nn.pytorch.conv.SAGEConv(in_feats=100, out_feats=100, aggregator_type='gcn', feat_drop=0.5)

        self.activation = torch.nn.ReLU()

        self.Linear = torch.nn.Linear(lstmout_dim * 2, class_num, bias=True)

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']
        # 如果出现单词表里卖弄不存在的单词，那么就返回UNK这个单词的id

        return result

    def load(self, corpus_file):
        model = {}
        file = open(corpus_file, encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            temp_list = line.split(' ')
            word = temp_list[0]
            vec = [float(x) for x in temp_list[1:-1]]
            vec.append(float(temp_list[-1].replace('\n', '')))
            model[word] = np.array(vec)
        return model

    def load_word2vec(self, word2vec_file):
        model = self.load(word2vec_file)

        embedding_matrix = []

        pad = [0] * 300

        embedding_matrix.append(pad)  # 手动在第一位添加PAD的全零词向量

        for word in self.vocab[1:]:  # 第0个是PAD，在glove里面没有
            # 它是根据vocab里面的顺序进行的词向量的构建
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                # print(word)
                embedding_matrix.append(model['the'])
            # 如果不存在单词那么就传入the，因为影响很小

        embedding_matrix = np.array(embedding_matrix)

        print(embedding_matrix)

        # embedding matrix就是指每条评论的词向量

        return embedding_matrix

    def add_all_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []

        local_vocab = list(set(doc_ids))

        for i, src_word_old in enumerate(local_vocab):
            src = old_to_new[src_word_old]
            for dst_word_old in local_vocab[i:]:
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            # self circle,自己的环
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def add_seq_edges(self, doc_ids: list, old_to_new: dict, dep: str):
        edges = []
        old_edge_id = []
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            # 把src的邻居节点加入
            for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]

                # - first connect the new sub_graph
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])


            # self circle
            # 加入src的自环
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        # TODO：接下来就是把有依存关系的词加进来，方法跟上列方法一致
        try:
            for depen in dep.split(' '):
                # src_index表示在句内的索引，得到单词id
                dependency = depen.split('-')
                src_index = int(dependency[0])
                dst_index = int(dependency[1])
                if src_index >= len(doc_ids) - 2 or dst_index >= len(doc_ids) - 2:
                    continue
                src_word_old = doc_ids[src_index]
                dst_word_old = doc_ids[dst_index]
                src = old_to_new[src_word_old]
                dst = old_to_new[dst_word_old]

                # - first connect the new sub_graph
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
        except:
            pass

        return edges, old_edge_id

    def lstm_feature(self, embedding):
        context, _ = self.bilstm(embedding)
        return context

    def gcn_feature(self, g, feat):
        g = dgl.add_self_loop(g)
        temp1 = F.relu(self.gcn1(g, feat))
        temp2 = F.relu(self.gcn2(g, temp1))
        context = F.relu(self.gcn3(g, temp2))
        return context

    def sage_feature(self, g, feat):
        g = dgl.add_self_loop(g)
        temp1 = F.relu(self.sage1(g, feat))
        temp2 = F.relu(self.sage2(g, temp1))
        context = F.relu(self.sage3(g, temp2))
        return context

    def regulate(self, doc_ids):
        new = []
        length = []
        for doc in doc_ids:
            if len(doc) > self.max_length:
                temp = doc[:self.max_length]
            else:
                temp = doc
            length.append(len(temp))
            new.append(temp)
        part_max = max(length)
        return new, part_max

    def lstm_embedding(self,partmax,doc_ids: list):
        '''
        提取输入进LSTM的词向量
        '''
        if len(doc_ids) < partmax:
            doc_ids = list(keras.preprocessing.sequence.pad_sequences([doc_ids], padding="post", maxlen=partmax)[0])
            # 对长度不足的评论进行零填充

        # print('去重后数量：', len(doc_ids))
        # print('去重后数量(dgl结点数)：', len(list(set(doc_ids))))

        local_vocab = doc_ids[:]

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        local_node_hidden = self.node_hidden(local_vocab)

        # print('local_node_hidden:', local_node_hidden.shape, local_node_hidden)

        local_node_hidden.to('cuda:0')

        return local_node_hidden

    def new_map(self, word_ids, feat):
        '''
        得到按照语序去重后的单词列表和对应的词向量
        '''
        map_dict = {}
        uniq_list = []
        for word in word_ids:
            if word in map_dict.keys():
                pass
            else:
                uniq_list.append(word)
                map_dict[word] = word_ids.index(word)
        graph_feat = []
        for w in uniq_list:
            graph_feat.append(feat[map_dict[w]])

        # 不知道经过二次提取之后grad还在不在, 先试一下效果

        return uniq_list, graph_feat

    def seq_to_graph(self, doc_ids: list, features, dep: str) -> dgl.DGLGraph().to('cuda:0'):
        # 可以试试把PAD这个节点去掉怎么样

        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]

        # local_node_hidden_all = self.node_hidden(torch.tensor(doc_ids).to('cuda:0'))

        # print('doc_ids', len(doc_ids))
        # print('local_node_hidden_all', local_node_hidden_all.shape, local_node_hidden_all)

        local_vocab, gin_feats = self.new_map(doc_ids, features)  # 得到去重后的单词列表和相应的经过LSTM训练后的词向量

        # local_vocab = set(doc_ids)

        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph_temp = dgl.DGLGraph()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 修改
        sub_graph = sub_graph_temp.to('cuda:0')

        # https://docs.dgl.ai/guide_cn/graph-gpu.html

        sub_graph.add_nodes(len(local_vocab.to('cuda:0')))
        # local_node_hidden = self.node_hidden(local_vocab)

        # print('local_vocab', len(local_vocab))
        # print('local_node_hidden',local_node_hidden.shape,local_node_hidden)
        # local_node_hidden.to('cuda:0')

        # in_feat = torch.stack(gin_feats, dim=0).cuda()
        #
        # gin_feat = self.sage_feature(sub_graph, in_feat)
        #
        # # print('gin_feat', gin_feat.size(), gin_feat.grad_fn)
        #
        # sub_graph.ndata['h'] = gin_feat.to('cuda:0')

        seq_edges, seq_old_edges_id = self.add_seq_edges(doc_ids, old_to_new, dep)

        edges, old_edge_id = [], []
        # edges = []

        edges.extend(seq_edges)

        old_edge_id.extend(seq_old_edges_id)

        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)
        # 对边的权重进行更新

        srcs, dsts = zip(*edges)
        # srcs.to('cuda:0')
        # dsts.to('cuda:0')
        sub_graph.add_edges(srcs, dsts)

        try:
            seq_edges_w = self.seq_edge_w(old_edge_id)
        except RuntimeError:
            print(old_edge_id)

        seq_edges_w.to('cuda:0')

        sub_graph.edata['w'] = seq_edges_w
        # 这里才更新了边的权重，之前没有边的权重

        in_feat = torch.stack(gin_feats, dim=0).cuda()

        sub_graph.ndata['in_feat'] = in_feat.to('cuda:0')

        # sub_graph = dgl.add_self_loop(sub_graph)

        gin_feat = self.sage_feature(sub_graph, in_feat)

        # print('gin_feat', gin_feat.size(), gin_feat.grad_fn)

        sub_graph.ndata['h'] = gin_feat.to('cuda:0')

        return sub_graph

    def forward(self, docids, deps, is_20ng=None):  # 这里传入的是train的content， 评论词id 按照语序排序
        '''
        forword函数中传入的参数就是在model（）函数中传入的参数
        具体content的内容就是batch_iter里面进行的操作，是按照语序的顺序进行的操作
        '''
        doc_ids, partmax = self.regulate(docids)


        padded_inputs = keras.preprocessing.sequence.pad_sequences(doc_ids, padding="post",
                                                                   maxlen=partmax)

        mask = torch.tensor(padded_inputs != 0).to('cuda:0')

        mask.to('cuda:0')

        embed = torch.stack([self.lstm_embedding(partmax, doc) for doc in doc_ids], dim=0)
        # emdeb-->tensor

        embed = embed * mask.unsqueeze(2).float().expand_as(embed).to('cuda:0')

        embed.to('cuda:0')

        # print('embed', embed.size(), embed.grad_fn)

        lstm_features = self.lstm_feature(embed)

        # print('lstm_features', lstm_features.size(), lstm_features.grad_fn)

        # sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]

        sub_graphs = []

        for num in range(len(doc_ids)):
            doc = doc_ids[num]

            features = lstm_features[num]

            dep = deps[num]

            # print('features',features.size())

            sub = self.seq_to_graph(doc, features, dep)

            sub_graphs.append(sub)

        batch_graph = dgl.batch(sub_graphs)

        # 消息传播机制
        # batch_graph.update_all(
        #     message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
        #     reduce_func=dgl.function.max('weighted_message', 'h')
        # )

        # 消息传播机制是为了把各个节点和边的权重进行相乘，但消息传播机制有可能会让效果变差

        # 对向量进行加和，因为一句话的情感由所有单词共同决定，所以先将所有的词向量加和处理
        h1 = dgl.sum_nodes(batch_graph, feat='h')

        drop1 = self.dropout(h1)
        # 这里是softmax层，最后输出最后的可能性
        act1 = self.activation(drop1)

        l = self.Linear(act1)

        return l
