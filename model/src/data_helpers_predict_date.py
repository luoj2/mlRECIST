import numpy as np
import re
import xlrd
from xlrd import open_workbook
import datetime
from datetime import date

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

def read_information(data_file):
    wb = open_workbook(data_file)
    s = wb.sheet_by_index(1)
    data = {}
    ids = []
    first_progression = {}
    for row in range(1,s.nrows):
        id = s.cell(row,0).value
        if id not in data:
            data[id] = []
            ids.append(id)
        if id not in first_progression:
            first_progression[id] = ''
        row_info = {}
        row_info['id'] = id
        row_info['progress'] = s.cell(row,5).value
        row_info['type'] = s.cell(row,6).value
        if (s.cell(row,6).value == 'progression') and (s.cell(row,5).value == 1) and first_progression[id] == '':
            first_progression[id] = s.cell(row,8).value
        row_info['incl'] = s.cell(row,7).value
        row_info['scan_date'] = s.cell(row,8).value
        row_info['act_scan_date'] = xlrd.xldate.xldate_as_datetime(row_info['scan_date'], wb.datemode)
#         row_info['text'] = s.cell(row,11).value
        row_info['text'] = refine(s.cell(row,11).value)
        data[id].append(row_info)
    for id in ids:
        for row_info in data[id]:
            if first_progression[id] != '':
                row_info['progress_date'] = first_progression[id]
                row_info['act_progress_date'] = xlrd.xldate.xldate_as_datetime(first_progression[id], wb.datemode)
            else:
                row_info['progress_date'] = ''
                row_info['act_progress_date'] = ''
    return data, ids

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

def parse_data(ids, data):
    pre_treatments = []
    post_treatments = []
    labels = []
    ori_id = []
    ori_lbl = []
    scan_dates = []
    pro_dates = []
    
    for id in ids:
#         print "*****************************"
#         print "*****************************"
#         print "*****************************"
#         print('ID is" {}'.format(id))
        pre = ''
        progress_day = ''
        post = {}
        act_sd = {}
        act_pd = {}
        for record in data[id]:
#             print('Progress is" {}'.format(record['progress']))
            if record['incl'] == 'no':
                continue
            if record['progress'] == 1:
                progress_day = record['progress_date']
            if record['type'] == 'baseline':
                txt = ' '.join(record['text'].split())
                pre += ' ' + txt
            else:
                if record['scan_date'] not in post:
                    post[record['scan_date']] = ''
                txt = ' '.join(record['text'].split())
                post[record['scan_date']] += ' ' + txt
                act_sd[record['scan_date']] = record['act_scan_date']
                act_pd[record['scan_date']] = record['act_progress_date']
#         print("Progress date is: {}".format(progress_day))
        if progress_day == '':
            for post_date in post:
                lbl = 'no'
                pre_treatments.append(pre)
                post_treatments.append(post[post_date])
                labels.append(lbl)
                ori_id.append(id)
                ori_lbl.append('no')
                scan_dates.append(act_sd[post_date])
                pro_dates.append(act_pd[post_date])
#                 print "--------------"
#                 print(lbl)
#                 print(pre)
#                 print(post[post_date])
        else:
            for post_date in post:
                if post_date == progress_day:
                    lbl = 'yes'
                    pre_treatments.append(pre)
                    post_treatments.append(post[post_date])
                    labels.append(lbl)
                    ori_id.append(id)
                    ori_lbl.append('yes')
                    scan_dates.append(act_sd[post_date])
                    pro_dates.append(act_pd[post_date])
#                     print "--------------"
#                     print(lbl)
#                     print(pre)
#                     print(post[post_date])
                else:
                    lbl = 'no'
                    pre_treatments.append(pre)
                    post_treatments.append(post[post_date])
                    labels.append(lbl)
                    ori_id.append(id)
                    ori_lbl.append('yes')
                    scan_dates.append(act_sd[post_date])
                    pro_dates.append(act_pd[post_date])
#                     print "--------------"
#                     print(lbl)
#                     print(pre)
#                     print(post[post_date])
        
    return pre_treatments , post_treatments, labels, ori_id, ori_lbl, scan_dates, pro_dates

def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    data, ids = read_information(data_file)
    ids = np.array(list(ids))
        
    pre_treatments , post_treatments, labels, ori_id, ori_lbl, scan_dates, pro_dates = parse_data(ids, data)
    x_pre = [s.lower() for s in pre_treatments]
    x_post = [s.lower() for s in post_treatments]
    y = []
    for lbl in labels:
        if lbl == 'yes':
            y.append(map(np.float32,[1,0]))
        else:
            y.append(map(np.float32,[0,1]))
    y = np.array(y)
    
    return [x_pre, x_post, y, ori_id, ori_lbl, scan_dates, pro_dates]


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