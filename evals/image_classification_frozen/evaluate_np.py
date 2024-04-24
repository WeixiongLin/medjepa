from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
# from Dataset.dataset import RadNet_Dataset_early, RadNet_Dataset_late
# from Model.modelRadNet import RadNet
import numpy as np
import torch
# from My_Trainer.trainer import Trainer
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef
from functools import partial
import wandb
import sys
import random
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from transformers import AutoModel,BertConfig,AutoTokenizer
from safetensors.torch import load_model
import json


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def find_nearest(array, value):
    """找到数组中最接近给定值的元素的索引"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calculate_multilabel_AUC(labels, logits):
    # """计算多标签分类的平均AUC"""
    # 初始化AUC列表
    auc_list = []
    fpr_list = []
    tpr_list = []
    scores = sigmoid(logits)
    # logits = np.exp(logits)
    # 遍历每一列（即每个类别）
    for i in range(labels.shape[1]):
        # 计算当前类别的AUC
        # 注意：roc_auc_score 预期的输入是二进制标签和概率或者分数
        # logits 需要通过sigmoid函数转换成概率
        if len(np.unique(labels[:, i])) < 2:
            continue
        auc = roc_auc_score(labels[:, i], scores[:, i])
        auc_list.append(auc)

        fpr, tpr, _ = roc_curve(labels[:, i], scores[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    # 计算所有类别AUC的平均值
    max_auc = np.max(auc_list)
    mean_auc = np.mean(auc_list)
    min_auc = np.min(auc_list)
    max_index = auc_list.index(max_auc)
    min_index = auc_list.index(min_auc)

    #每个sample都可能会有多个标签，寻找每个sample里面的prob最大的10，20，30个index，看看这些index和标签的交集所占标签总数的比例

    return mean_auc, max_auc, max_index, min_auc, min_index



def calculate_mAP(labels, logits):
    """
    计算多标签分类的平均精度(mAP)。
    param labels:基于真值标签的二元矩阵(形状[B, cls])
    param logits:预测分数或logits的矩阵(shape [B, cls])
    return: mAP分数
    """
    # 类别数量
    # scores = np.exp(logits)
    scores = sigmoid(logits)
    n_classes = labels.shape[1]
    
    # 初始化AP_list
    AP_list = []
    precision_list = []
    recall_list = []
    
    # 对每个类别计算一个AP
    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels[:, class_i], scores[:, class_i])
        # Calculate average precision for this class
        AP = average_precision_score(labels[:, class_i], scores[:, class_i])
        AP_list.append(AP)
        precision_list.append(precision)
        recall_list.append(recall)
    
    # 计算所有类别的AP的平均
    mAP = np.mean(AP_list)
    max_ap = np.max(AP_list)
    min_ap = np.min(AP_list)
    max_index = AP_list.index(max_ap)
    min_index = AP_list.index(min_ap)
    
    return mAP

def calculate_mF1max_MCC(labels, logits):
    scores = sigmoid(logits)
    n_classes = labels.shape[1]

    F1max_list = []
    MCC_list = []

    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        precision, recall, thresholds = precision_recall_curve(labels[:, class_i], scores[:, class_i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0 
        max_f1_index = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_index]
        max_f1_threshold = thresholds[max_f1_index]
        F1max_list.append(max_f1)
        pred_class = (scores[:, class_i] >= max_f1_threshold).astype(int)
        mcc = matthews_corrcoef(labels[:, class_i], pred_class)
        MCC_list.append(mcc)

    return np.mean(F1max_list), np.mean(MCC_list)

def calculate_mRecall_FPR(labels, logits, fpr_points):
    scores = sigmoid(logits)
    n_classes = labels.shape[1]

    macro_recall_at_fpr = {fpr: [] for fpr in fpr_points}
    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(labels[:, class_i], scores[:, class_i])

        for point in fpr_points:
            idx = find_nearest(fpr, point)
            macro_recall_at_fpr[point].append(tpr[idx])

    average_recall_at_fpr = {fpr: np.mean(recalls) for fpr, recalls in macro_recall_at_fpr.items()}
    return average_recall_at_fpr

