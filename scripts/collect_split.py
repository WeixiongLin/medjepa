"""
由于 @xiaoman 和 @qiaoyu 分别给出了对于 radiopedia dataset 的 split, 并且数据的表现形式不同。
我们需要分别对于 @xiaoman @qiaoyu 的 data split 进行统计, 根据 npy_path 判断两者的 split 是否相同。

cd /mnt/petrelfs/linweixiong/jepa/scripts &&
python collect_split.py
"""

import json
from tqdm import tqdm
import os

ROOT_PATH = '/mnt/petrelfs/linweixiong'


################################################################################
# 1. @xiaoyman 的 data split
################################################################################
def collect_qiaoyu_dataset(file_path):
    npy_list = []
    with open(file_path) as f:
        data_dict = json.load(f)

    for source_url in data_dict:
        item = data_dict[source_url]["Samples"]
        for rsample_id in range( len(item) ):
            sample = item[rsample_id]
            npy_path = sample['npy_path']
            npy_list.append(npy_path)
            # raise RuntimeError( npy_path )
        # end for
    # end for

    return npy_list
# end def

# len(npy_list) = 43094, npy_list[0] = 3/6707.npy
file_path = f'{ROOT_PATH}/jepa/data/joy/radio_3d_case_level_link_dict_final_all_new_train.json'
qiaoyu_npy_list = collect_qiaoyu_dataset(file_path)
qiaoyu_npy_set = set(qiaoyu_npy_list)

# (43094, 43085)
# raise RuntimeError( len(qiaoyu_npy_list), len(qiaoyu_npy_set) )


################################################################################
# 2. @xiaoyman 的 data split
################################################################################

def collect_xiaoman_dataset(file_path):
    npy_list = []
    with open(file_path) as f:
        data_list = json.load(f)

    for item_id in tqdm(range(len(data_list))):
        item = data_list[item_id]
        npy_path = item['npy_path']
        npy_list.append(npy_path)
    # end for

    return npy_list
# end def


# len(npy_list) = 2893, npy_list[0] = 0/0.npy
# file_path = f'{ROOT_PATH}/jepa/data/radiology_article_npy_test.json'

# len(npy_list) = 47883, npy_list[0] = 0/3682.npy
file_path = f'{ROOT_PATH}/jepa/data/radiology_article_npy_train.json'
xiaoman_npy_list = collect_xiaoman_dataset(file_path)
xiaoman_npy_set = set(xiaoman_npy_list)

# (47883, 47883)
# raise RuntimeError( len(xiaoman_npy_list), len(xiaoman_npy_set) )

shared_npy = set.intersection(qiaoyu_npy_set, xiaoman_npy_set)

print( len(xiaoman_npy_set), len(qiaoyu_npy_set), len(shared_npy) )
