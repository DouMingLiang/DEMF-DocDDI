
# binary classification task

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.utils.data as Data
from allennlp.modules.matrix_attention import DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from gensim.models import KeyedVectors
import linecache
import re
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize
import torch.optim as optim
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertModel, BertTokenizer, BertConfig
import os
import random
from attn_unet import AttentionUNet

config_path = '../BioBERT_pytorch/config.json'
model_path = '../BioBERT_pytorch'
vocab_path = '../BioBERT_pytorch/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_path)

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
sequence_length = 512
BATCH_SIZE = 1

article_text = []
article_drug = []
article_pairs = []

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# Data loading
def load_data(root_Data):
    article_load_text = []
    article_load_drug = []
    article_load_pairs = []
    pairs_sum_max = 0

    total_line = len(open(root_Data).readlines())

    for i in range(total_line):
        pairs_sum = 0

        line = linecache.getline(root_Data, i + 1)
        line = line.strip()
        list_1 = [j.start() for j in re.finditer('==', line)]
        list_2 = [j.start() for j in re.finditer('\$', line)]
        sentence_raw = line[list_1[2] + 2: list_2[2]]
        article_load_text.append(sentence_raw)

        drug_raw = line[list_1[3] + 1: list_2[3]]
        list_3 = [j.start() for j in re.finditer('\{', drug_raw)]
        list_4 = [j.start() for j in re.finditer('\}', drug_raw)]

        drug_curr = []

        for j in range(0, len(list_3)):
            drug_curr_1 = []
            drug_raw_1 = drug_raw[list_3[j] + 1: list_4[j]]

            list_5 = [k.start() for k in re.finditer(';', drug_raw_1)]
            drug_name_raw = drug_raw_1[0: list_5[0]]
            drug_curr_1.append(drug_name_raw)

            drug_raw_2 = drug_raw_1[list_5[0] + 1:]

            list_6 = [k.start() for k in re.finditer('\[', drug_raw_2)]
            list_7 = [k.start() for k in re.finditer('\]', drug_raw_2)]
            list_8 = [k.start() for k in re.finditer(', ', drug_raw_2)]
            drug_curr_2 = []

            if len(list_8) == 0:
                drug_pos_int = int(drug_raw_2[list_6[0] + 1: list_7[0]])
                drug_curr_2.append(drug_pos_int)
            else:
                for q in range(0, len(list_8) + 1):
                    if q == 0:
                        drug_pos_int = int(drug_raw_2[list_6[0] + 1: list_8[0]])
                    elif q == len(list_8):
                        drug_pos_int = int(drug_raw_2[list_8[q - 1] + 1: list_7[0]])
                    else:
                        drug_pos_int = int(drug_raw_2[list_8[q - 1] + 1: list_8[q]])

                    drug_curr_2.append(drug_pos_int)

            drug_curr_1.append(drug_curr_2)
            drug_curr.append(drug_curr_1)

        article_load_drug.append(drug_curr)

        pairs_raw = line[list_1[4] + 1: list_2[4]]
        list_9 = [k.start() for k in re.finditer('\{', pairs_raw)]
        list_10 = [k.start() for k in re.finditer('\}', pairs_raw)]

        pairs_curr = []

        pairs_sum = pairs_sum + len(list_9)
        if pairs_sum > pairs_sum_max:
            pairs_sum_max = pairs_sum

        for j in range(0, len(list_9)):
            pairs_raw_1 = pairs_raw[list_9[j] + 1: list_10[j]]
            list_11 = [k.start() for k in re.finditer(';', pairs_raw_1)]
            drug_1_raw = pairs_raw_1[0: list_11[0]]
            drug_2_raw = pairs_raw_1[list_11[0] + 1: list_11[1]]
            lable_raw = pairs_raw_1[list_11[1] + 1: list_11[2]]
            overlapNum_lable_false = pairs_raw_1[list_11[2] + 1: list_11[3]]
            overlapNum_lable_true = pairs_raw_1[list_11[3] + 1:]

            lable_curr = []
            lable_curr.append(drug_1_raw)
            lable_curr.append(drug_2_raw)
            lable_curr.append(lable_raw)
            lable_curr.append(overlapNum_lable_false)
            lable_curr.append(overlapNum_lable_true)
            pairs_curr.append(lable_curr)

        article_load_pairs.append(pairs_curr)

    return article_load_text, article_load_drug, article_load_pairs, pairs_sum_max

def attention_masks(ids):
    id_mask = [int(i > 0) for i in ids]  # PAD: 0; Or: 1
    return id_mask

# Data processing
def data_processing(article_load_text, article_load_drug, article_load_pairs):

    article_text_pro = []
    article_text_pro_token = []
    article_text_pro_id = []
    article_text_pro_mask = []
    article_pairs_num = []

    for i in range(0, len(article_load_text)):
        real_str_NUM = re.sub(r'\[CLS\] ', '', article_load_text[i])
        real_str_NUM = re.sub(r' \[SEP\]', '', real_str_NUM)
        article_text_pro.append(real_str_NUM)

    drug_pos = []
    for i in range(0, len(article_text_pro)):
        article_token_curr = article_load_text[i]
        article_token_curr = tokenizer.tokenize(article_token_curr)
        article_text_pro_token.append(article_token_curr)
        drug_pos_article = []

        pos_1 = []
        pos_2 = []

        for j in range(0, len(article_token_curr)):
            if article_token_curr[j] == '[':
                pos_1.append(j)
            if article_token_curr[j] == ']':
                pos_2.append(j)

        for j in range(0, len(pos_1)):
            curr_pos_num = []
            if (pos_2[j] - pos_1[j]) == 2:
                pos_len = 1
                curr_pos_num.append(pos_len)
                curr_pos_num.append(pos_1[j] + 1)
            else:
                pos_len = pos_2[j] - pos_1[j] - 1
                curr_pos_num.append(pos_len)
                for k in range(pos_1[j] + 1, pos_2[j]):
                    curr_pos_num.append(k)
            drug_pos_article.append(curr_pos_num)

        pos_curr = []
        for j in range(0, len(article_load_drug[i])):
            drug_name_curr = article_load_drug[i][j][0]
            pos_curr_1 = []
            pos_curr_2 = []
            for k in range(0, len(article_load_drug[i][j][1])):
                pos_num = article_load_drug[i][j][1][k]
                pos_token = drug_pos_article[pos_num]
                pos_curr_1.append(pos_token)
            pos_curr_2.append(drug_name_curr)
            pos_curr_2.append(pos_curr_1)
            pos_curr.append(pos_curr_2)
        drug_pos.append(pos_curr)

    pairs_lable = []
    pairs_false_overNum = []
    pairs_true_overNum = []
    pairs_drug1_pos = []
    pairs_drug2_pos = []
    pairs_drug_pos = []
    hts = []

    hts_curr_1 = None
    hts_curr_2 = None
    for i in range(0, len(article_load_pairs)):
        article_pairs_num.append(len(article_load_pairs[i]))
        pairs_lable_curr = [0] * pairs_num_max
        pairs_false_overNum_curr = [0] * pairs_num_max
        pairs_true_overNum_curr = [0] * pairs_num_max

        drug_pos_curr_1 = []
        drug_pos_curr_2 = []
        drug_pos_curr = []
        hts_curr_N = []
        hts_curr = []
        for j in range(0, len(article_load_pairs[i])):
            lable = article_load_pairs[i][j][2]
            if lable.lower() == 'false':
                pairs_lable_curr[j] = 1
            elif lable.lower() == 'true':
                pairs_lable_curr[j] = 2

            pairs_false_overNum_curr[j] = int(article_load_pairs[i][j][3])
            pairs_true_overNum_curr[j] = int(article_load_pairs[i][j][4])

            drug_1_name = article_load_pairs[i][j][0]
            drug_2_name = article_load_pairs[i][j][1]

            for entity_num, e in enumerate(drug_pos[i]):
                if hts_curr_1 is None and (e[0] == drug_1_name):
                    hts_curr_1 = entity_num
                if hts_curr_2 is None and (e[0] == drug_2_name):
                    hts_curr_2 = entity_num
            hts_curr.append(hts_curr_1)
            hts_curr.append(hts_curr_2)
            hts_curr_N.append(hts_curr)
            hts_curr_1 = None
            hts_curr_2 = None
            hts_curr = []

            for k in range(0, len(drug_pos[i])):
                if drug_pos[i][k][0] == drug_1_name:
                    drug_pos_curr_1.append(drug_pos[i][k][1])
                if drug_pos[i][k][0] == drug_2_name:
                    drug_pos_curr_2.append(drug_pos[i][k][1])
        for k in range(0, len(drug_pos[i])):
            drug_pos_curr.append(drug_pos[i][k][1])

        pairs_lable.append(pairs_lable_curr)
        pairs_false_overNum.append(pairs_false_overNum_curr)
        pairs_true_overNum.append(pairs_true_overNum_curr)
        pairs_drug1_pos.append(drug_pos_curr_1)
        pairs_drug2_pos.append(drug_pos_curr_2)
        pairs_drug_pos.append(drug_pos_curr)
        hts.append(hts_curr_N)

    for i in range(0, len(article_text_pro_token)):
        id_curr = [[0 for col in range(512)] for row in range(5)]  # [[0]*512]*5
        mask_curr = [[0 for col in range(512)] for row in range(5)]
        article_text_pro_id_curr = tokenizer.encode(article_text_pro_token[i])
        article_len_curr = len(article_text_pro_id_curr)

        for j in range(0, article_len_curr):
            if j < 512:
                id_curr[0][j] = article_text_pro_id_curr[j]
                mask_curr[0][j] = 1
            elif (512 <= j) and (j < 1024):
                id_curr[1][j - 512] = article_text_pro_id_curr[j]
                mask_curr[1][j - 512] = 1
            elif (1024 <= j) and (j < 1536):
                id_curr[2][j - 1024] = article_text_pro_id_curr[j]
                mask_curr[2][j - 1024] = 1
            elif (1536 <= j) and (j < 2048):
                id_curr[3][j - 1536] = article_text_pro_id_curr[j]
                mask_curr[3][j - 1536] = 1
            elif (2048 <= j) and (j < 2560):
                id_curr[4][j - 2048] = article_text_pro_id_curr[j]
                mask_curr[4][j - 2048] = 1

        article_text_pro_id.append(id_curr)
        article_text_pro_mask.append(mask_curr)

    return article_text_pro_id, article_text_pro_mask, pairs_lable, \
           pairs_false_overNum, pairs_true_overNum, pairs_drug1_pos, pairs_drug2_pos, article_pairs_num, hts, pairs_drug_pos

def gen_dataloader():

    article_id_seq_ten = torch.LongTensor(article_id_seq) # article_num*5*512
    article_mask_seq_ten = torch.LongTensor(article_mask_seq) # article_num*5*512
    drug_pairs_label_ten = torch.LongTensor(drug_pairs_label) # article_num*pairs_num_max
    pairs_false_overNum_ten = torch.LongTensor(pairs_false_overNum)  # article_num*pairs_num_max
    pairs_true_overNum_ten = torch.LongTensor(pairs_true_overNum)

    torch_dataset = Data.TensorDataset(article_id_seq_ten, article_mask_seq_ten, drug_pairs_label_ten, pairs_false_overNum_ten, pairs_true_overNum_ten)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    return loader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        kernel_sizes = [5]
        dropout = 0.5
        flag_class = 5
        hidden_size = 384

        self.conv_list = nn.ModuleList([nn.Conv1d(hidden_size * 2,
                                                  hidden_size, w, padding=(w - 1) // 2) for w in kernel_sizes])

        self.classifier = nn.Linear(len(kernel_sizes) * hidden_size, flag_class)
        self.dropout = nn.Dropout(dropout)

        self.segmentation_net = AttentionUNet(input_channels=3,
                                              class_number=256,
                                              down_channel=256)
        self.UNet_extractor = nn.Linear(256, 64)
        self.bilinear = nn.Linear(832 * 64, 2)

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)

        return htss

    def forward(self, ids_in, masks_in, article_num):
        ids_in = ids_in.squeeze(0)
        masks_in = masks_in.squeeze(0)

        for i in range(0, 5):
            if i == 0:
                input_id_0 = ids_in[i]  # torch.Size([512])
                input_mask_0 = masks_in[i]  # torch.Size([512])
                input_id_0 = input_id_0.unsqueeze(0)  # torch.Size([1, 512])
                input_mask_0 = input_mask_0.unsqueeze(0)  # torch.Size([1, 512])
                out_curr = self.bert(input_id_0, attention_mask=input_mask_0)
                out_0 = out_curr[0]  # torch.Size([1, 512, 768])
                out_0 = out_0.squeeze(0)  # torch.Size([512, 768])
            elif i == 1:
                input_id_1 = ids_in[i]  # torch.Size([512])
                input_mask_1 = masks_in[i]  # torch.Size([512])
                input_id_1 = input_id_1.unsqueeze(0)  # torch.Size([1, 512])
                input_mask_1 = input_mask_1.unsqueeze(0)  # torch.Size([1, 512])
                out_curr = self.bert(input_id_1, attention_mask=input_mask_1)
                out_1 = out_curr[0]  # torch.Size([1, 512, 768])
                out_1 = out_1.squeeze(0)  # torch.Size([512, 768])
            elif i == 2:
                input_id_2 = ids_in[i]  # torch.Size([512])
                input_mask_2 = masks_in[i]  # torch.Size([512])
                input_id_2 = input_id_2.unsqueeze(0)  # torch.Size([1, 512])
                input_mask_2 = input_mask_2.unsqueeze(0)  # torch.Size([1, 512])
                out_curr = self.bert(input_id_2, attention_mask=input_mask_2)
                out_2 = out_curr[0]  # torch.Size([1, 512, 768])
                out_2 = out_2.squeeze(0)  # torch.Size([512, 768])
            elif i == 3:
                input_id_3 = ids_in[i]  # torch.Size([512])
                input_mask_3 = masks_in[i]  # torch.Size([512])
                input_id_3 = input_id_3.unsqueeze(0)  # torch.Size([1, 512])
                input_mask_3 = input_mask_3.unsqueeze(0)  # torch.Size([1, 512])
                out_curr = self.bert(input_id_3, attention_mask=input_mask_3)
                out_3 = out_curr[0]  # torch.Size([1, 512, 768])
                out_3 = out_3.squeeze(0)  # torch.Size([512, 768])
            elif i == 4:
                input_id_4 = ids_in[i]  # torch.Size([512])
                input_mask_4 = masks_in[i]  # torch.Size([512])
                input_id_4 = input_id_4.unsqueeze(0)  # torch.Size([1, 512])
                input_mask_4 = input_mask_4.unsqueeze(0)  # torch.Size([1, 512])
                out_curr = self.bert(input_id_4, attention_mask=input_mask_4)
                out_4 = out_curr[0]  # torch.Size([1, 512, 768])
                out_4 = out_4.squeeze(0)  # torch.Size([512, 768])

        drug_total_emb_m1 = torch.zeros(len(drug_pos[article_num]), 768)
        drug_total_emb_1 = torch.zeros(len(drug_1_pos[article_num]), 768)
        drug_total_emb_2 = torch.zeros(len(drug_2_pos[article_num]), 768)

        for i in range(0, len(drug_pos[article_num])):

            drug_emb_1 = torch.zeros(len(drug_pos[article_num][i]), 768)  # same drug_num * 768
            drug_emb_curr_1 = torch.zeros(768)

            for j in range(0, len(drug_pos[article_num][i])):
                drug_emb_a = torch.zeros(len(drug_pos[article_num][i][j]) - 1, 768)  # drug_pos_len * 768

                for k in range(1, drug_pos[article_num][i][j][0] + 1):
                    if drug_pos[article_num][i][j][k] < 512:
                        drug_emb_a[k - 1] = out_0[drug_pos[article_num][i][j][k]]
                    elif (drug_pos[article_num][i][j][k] >= 512) and (drug_pos[article_num][i][j][k] < 1024):
                        drug_emb_a[k - 1] = out_1[drug_pos[article_num][i][j][k] - 512]
                    elif (drug_pos[article_num][i][j][k] >= 1024) and (drug_pos[article_num][i][j][k] < 1536):
                        drug_emb_a[k - 1] = out_2[drug_pos[article_num][i][j][k] - 1024]
                    elif (drug_pos[article_num][i][j][k] >= 1536) and (drug_pos[article_num][i][j][k] < 2048):
                        drug_emb_a[k - 1] = out_3[drug_pos[article_num][i][j][k] - 1536]
                    elif (drug_pos[article_num][i][j][k] >= 2048) and (drug_pos[article_num][i][j][k] < 2560):
                        drug_emb_a[k - 1] = out_4[drug_pos[article_num][i][j][k] - 2048]

                drug_emb_curr_1 = torch.zeros(768)
                for k in range(0, len(drug_pos[article_num][i][j]) - 1):
                    drug_emb_curr_1 = drug_emb_curr_1 + drug_emb_a[k]

                drug_emb_curr_1 = drug_emb_curr_1 / (len(drug_pos[article_num][i][j]) - 1)
                drug_emb_1[j] = drug_emb_curr_1

            for j in range(0, len(drug_pos[article_num][i])):
                drug_emb_curr_1 = drug_emb_curr_1 + drug_emb_1[j]
            drug_emb_curr_1 = drug_emb_curr_1 / len(drug_pos[article_num][i])
            drug_total_emb_m1[i] = drug_emb_curr_1

        for i in range(0, len(hts[article_num])):
            drug_total_emb_1[i] = drug_total_emb_m1[hts[article_num][i][0]]

        for i in range(0, len(hts[article_num])):
            drug_total_emb_2[i] = drug_total_emb_m1[hts[article_num][i][1]]

        ent_encode = out_0.new_zeros(1, 120, 768)  # torch.Size([1, 42, 768])
        entity_emb = drug_total_emb_m1  # torch.Size([7, 768])
        entity_num = entity_emb.size(0)
        ent_encode[0, :entity_num, :] = entity_emb


        similar1 = DotProductMatrixAttention().cuda()(ent_encode, ent_encode).unsqueeze(-1)
        similar2 = CosineMatrixAttention().cuda()(ent_encode, ent_encode).unsqueeze(-1)
        similar3 = BilinearMatrixAttention(768, 768).cuda()(ent_encode, ent_encode).unsqueeze(-1)

        attn_input = torch.cat([similar1, similar2, similar3], dim=-1).permute(0, 3, 1, 2).contiguous()
        attn_map = self.segmentation_net(attn_input)
        hts_curr = [hts[article_num]]
        h_t_0 = self.get_ht(attn_map, hts_curr) # torch.Size([11, 256])
        h_t_1 = self.UNet_extractor(h_t_0) # torch.Size([11, 64])

        drug_total_emb_1 = drug_total_emb_1.cuda()
        drug_total_emb_2 = drug_total_emb_2.cuda()

        hs = torch.tanh(torch.cat([drug_total_emb_1, h_t_1], dim=1))  # torch.Size([11, 768+64])
        ts = torch.tanh(torch.cat([drug_total_emb_2, h_t_1], dim=1))  # torch.Size([11, 768+64])

        b1 = hs.view(-1, 832 // 64, 64)  # torch.Size([11, 13, 64])
        b2 = ts.view(-1, 832 // 64, 64)  # torch.Size([11, 13, 64])
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, 832 * 64)
        logits = self.bilinear(bl)

        return logits

if __name__ == '__main__':

    setup_seed(40)
    file_1 = open('results/GLDoc_DDI_2C.txt', 'w')
    file_name = 'train_DrugBank_doc_BC_FT.txt'
    pairs_num_max = 0
    article_text, article_drug, article_pairs, pairs_num_max = load_data(file_name)
    file_1.write(' In training, pairs_num_max = ' + str(pairs_num_max))

    article_id_seq = []
    article_mask_seq = []
    drug_pairs_label = []
    drug_1_pos = []
    drug_2_pos = []
    pairs_num = []
    pairs_false_overNum = []
    pairs_true_overNum = []

    hts = []
    drug_pos = []

    article_id_seq, article_mask_seq, drug_pairs_label, pairs_false_overNum, pairs_true_overNum, drug_1_pos, drug_2_pos, pairs_num, hts, drug_pos = \
        data_processing(article_text, article_drug, article_pairs)

    loader = gen_dataloader()

    model = Model()
    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, eps=1e-8)
    model.train()

    # Training
    for epoch in range(6):
        file_1.write('\nepoch: ' + str(epoch + 1))
        file_1.write('\n************')
        running_loss = 0
        sum_train = 0
        corrects_train = 0
        size_train = 0

        for i, (seq_ids, seq_masks, pairs_target, false_Num, true_Num) in enumerate(loader):
            optimizer.zero_grad()

            pairs_len = pairs_num[i]
            list_emp = [0] * pairs_len
            a = np.array(list_emp)
            pairs_label = torch.from_numpy(a)
            pairs_label = pairs_label.long()

            for j in range(0, pairs_len):
                pairs_label[j] = pairs_target[0][j] - 1

            pairs_label = pairs_label.cuda()
            out = model(seq_ids, seq_masks, i)
            corrects_train_cur = (torch.max(out, 1)[1].view(pairs_label.size()) == pairs_label).sum()
            corrects_train += corrects_train_cur
            size_train = size_train + pairs_len
            loss = criterion(out, pairs_label)
            loss.backward()

            optimizer.step()
            running_loss += float(loss.item())
            sum_train += 1

        accuracy_train = 100.0 * corrects_train / size_train
        loss_average = running_loss / sum_train
        file_1.write('\nepoch' + str(epoch + 1))
        file_1.write(' end, avergae loss is:' + str(loss_average))
        file_1.write(' acc = ' + str(accuracy_train))
    file_1.write('\n')

    # save model
    torch.save(model, "GLDoc_DDI_2C.pth")

    # Test
    corrects, avg_loss = 0, 0
    corrects_test = 0
    sum_test = 0

    file_name = 'test_DrugBank_doc_BC_FT.txt'
    pairs_num_max_1 = 0
    article_text, article_drug, article_pairs, pairs_num_max_1 = load_data(file_name)

    print('测试集中，pairs_num_max_1 = ', pairs_num_max_1)
    file_1.write(' In testing, pairs_num_max_1 = ' + str(pairs_num_max_1))

    article_id_seq = []
    article_mask_seq = []
    drug_pairs_label = []
    drug_1_pos = []
    drug_2_pos = []
    pairs_num = []
    pairs_false_overNum = []
    pairs_true_overNum = []
    hts = []
    drug_pos = []

    article_id_seq, article_mask_seq, drug_pairs_label, pairs_false_overNum, pairs_true_overNum, drug_1_pos, drug_2_pos, pairs_num, hts, drug_pos = \
        data_processing(article_text, article_drug, article_pairs)

    loader = gen_dataloader()

    model = torch.load("GLDoc_DDI_2C.pth")
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    for i, (seq_ids, seq_masks, pairs_target, false_Num, true_Num) in enumerate(loader):
        pairs_len = pairs_num[i]
        list_emp = [0] * pairs_len
        a = np.array(list_emp)
        pairs_label = torch.from_numpy(a)
        pairs_label = pairs_label.long()

        for j in range(0, pairs_len):
            pairs_label[j] = pairs_target[0][j] - 1

        pairs_label = pairs_label.cuda()
        with torch.no_grad():
            out_test = model(seq_ids, seq_masks, i)
        out_test_label = torch.max(out_test, 1)[1].view(pairs_label.size())

        if i == 0:
            out_test_matrix = out_test_label
            out_lable_matrix = pairs_label
        elif i > 0:
            out_test_matrix = torch.cat((out_test_matrix, out_test_label), 0)
            out_lable_matrix = torch.cat((out_lable_matrix, pairs_label), 0)

        pair_overlap_total = 0
        for j in range(0, pairs_len):
            pair_overlap_total = pair_overlap_total + false_Num[0][j] + true_Num[0][j]

        new_out_emp = [0] * pair_overlap_total
        d = np.array(new_out_emp)
        pairs_out_sen = torch.from_numpy(d)
        pairs_out_sen = pairs_out_sen.long()

        new_label_emp = [0] * pair_overlap_total
        e = np.array(new_label_emp)
        pairs_label_sen = torch.from_numpy(e)
        pairs_label_sen = pairs_label_sen.long()

        index_out = 0
        index_label = 0
        for j in range(0, pairs_len):
            if (false_Num[0][j] + true_Num[0][j]) == 1:
                pairs_out_sen[index_out] = out_test_label[j]
                pairs_label_sen[index_label] = pairs_label[j]
                index_out = index_out + 1
                index_label = index_label + 1
            elif (false_Num[0][j] + true_Num[0][j]) > 1:
                for k in range(0, false_Num[0][j] + true_Num[0][j]):
                    if k < false_Num[0][j]:
                        pairs_label_sen[index_label + k] = 0
                    else:
                        pairs_label_sen[index_label + k] = 1
                index_label = index_label + false_Num[0][j] + true_Num[0][j]

                if out_test_label[j] == pairs_label[j]:
                    for k in range(0, false_Num[0][j] + true_Num[0][j]):
                        if k < false_Num[0][j]:
                            pairs_out_sen[index_out + k] = 0
                        else:
                            pairs_out_sen[index_out + k] = 1
                else:
                    for k in range(0, false_Num[0][j] + true_Num[0][j]):
                        if k < true_Num[0][j]:
                            pairs_out_sen[index_out + k] = 0
                        else:
                            pairs_out_sen[index_out + k] = 1
                index_out = index_out + false_Num[0][j] + true_Num[0][j]

        if i == 0:
            out_test_matrix_sen = pairs_out_sen
            out_lable_matrix_sen = pairs_label_sen
        elif i > 0:
            out_test_matrix_sen = torch.cat((out_test_matrix_sen, pairs_out_sen), 0)
            out_lable_matrix_sen = torch.cat((out_lable_matrix_sen, pairs_label_sen), 0)

    y_pred_doc = out_test_matrix
    y_true_doc = out_lable_matrix

    y_pred_doc = y_pred_doc.cuda().cpu()
    y_true_doc = y_true_doc.cuda().cpu()

    micro_p = precision_score(y_true_doc, y_pred_doc, average='micro')
    micro_r = recall_score(y_true_doc, y_pred_doc, average='micro')
    micro_f1 = f1_score(y_true_doc, y_pred_doc, average='micro')
    file_1.write('\n document-level: micro_p = ' + str(micro_p) + 'micro_r' + str(micro_r) + 'micro_f1' + str(micro_f1))

    macro_p = precision_score(y_true_doc, y_pred_doc, average='macro')
    macro_r = recall_score(y_true_doc, y_pred_doc, average='macro')
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)
    file_1.write('\n document-level：macro_p = ' + str(macro_p) + 'macro_r' + str(macro_r) + 'macro_f1' + str(macro_f1))

    target_names = ['class 0', 'class 1']

    file_1.write('document-level：')
    file_1.write(str(classification_report(y_true_doc, y_pred_doc, target_names=target_names, digits=4)))

    y_pred_sen = out_test_matrix_sen
    y_true_sen = out_lable_matrix_sen

    y_pred_sen = y_pred_sen.cuda().cpu()
    y_true_sen = y_true_sen.cuda().cpu()

    micro_p_sen = precision_score(y_true_sen, y_pred_sen, average='micro')
    micro_r_sen = recall_score(y_true_sen, y_pred_sen, average='micro')
    micro_f1_sen = f1_score(y_true_sen, y_pred_sen, average='micro')

    file_1.write('\n sentence-level： micro_p = ' + str(micro_p_sen) + 'micro_r' + str(micro_r_sen) + 'micro_f1' + str(
        micro_f1_sen))

    macro_p_sen = precision_score(y_true_sen, y_pred_sen, average='macro')
    macro_r_sen = recall_score(y_true_sen, y_pred_sen, average='macro')
    macro_f1_sen = (2 * macro_p_sen * macro_r_sen) / (macro_p_sen + macro_r_sen)

    file_1.write('\n sentence-level：macro_p = ' + str(macro_p_sen) + 'macro_r' + str(macro_r_sen) + 'macro_f1' + str(
        macro_f1_sen))

    target_names = ['class 0', 'class 1']
    file_1.write('sentence-level：')
    file_1.write(str(classification_report(y_true_sen, y_pred_sen, target_names=target_names, digits=4)))
    file_1.close()