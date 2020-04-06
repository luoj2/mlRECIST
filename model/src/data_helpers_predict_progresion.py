import numpy as np
import re
from xlrd import open_workbook


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def refine(text):
    refine_text = ''
    flag = 0
    for line in text.splitlines():
        if line.startswith('FINDINGS:'):
            flag = 1
            continue
        elif line.startswith('Dictated By'):
            break
        elif flag == 1:
            refine_text += line + '\n'
    return refine_text

def read_information(data_file):
    wb = open_workbook(data_file)
    s = wb.sheet_by_index(1)
    data = {}
    for row in range(1,s.nrows):
        id = s.cell(row,0).value
        if id not in data:
            data[id] = []
        row_info = {}
        row_info['id'] = id
        row_info['progression'] = s.cell(row,5).value
        row_info['type'] = s.cell(row,6).value
        row_info['incl'] = s.cell(row,7).value
        row_info['text'] = refine(s.cell(row,11).value)
        data[id].append(row_info)
    
    pre_treatments = []
    post_treatments = []
    labels = []
    ori_id = []
    
    for id in data:
        pre = ''
        post = ''
        lbl = ''
        for record in data[id]:
            if record['incl'] == 'no':
                continue
            if record['type'] == 'baseline':
                txt = ' '.join(record['text'].split())
                pre += ' ' + txt
            elif record['type'] == 'ontx' or record['type'] == 'progression':
                txt = ' '.join(record['text'].split())
                post += ' ' + txt
            lbl = record['progression']
        pre_treatments.append(pre)
        post_treatments.append(post)
        labels.append(lbl)
        ori_id.append(id)
        
    return pre_treatments , post_treatments, labels, ori_id

def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    pre_treatments , post_treatments, labels, ori_id = read_information(data_file)
    x_pre = [s.lower() for s in pre_treatments]
    x_post = [s.lower() for s in post_treatments]
    y = []
    for lbl in labels:
        if lbl == 1:
            y.append(map(np.float32,[1,0]))
        else:
            y.append(map(np.float32,[0,1]))
    
    y = np.array(y)
    return [x_pre, x_post, y, ori_id]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]