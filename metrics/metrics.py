from typing import List, Any

import numpy as np


def user_hitrate(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Hitrate@k
    -------------------------------
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: 1 if top-k recommendations contains at lease one relevant item
    """
    return int(len(set(y_rec[:k]).intersection(set(y_rel))) > 0)


def user_precision(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Precision@k
    --------------------------------
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of relevant items through recommendations
    """
    precision = len(set(y_rec[:k]).intersection(set(y_rel)))/k # len(y_rec[:k])
    return precision


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Recall@k
    ----------------------------------
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of found relevant items through recommendations
    """
    recall = len(set(y_rec[:k]).intersection(set(y_rel)))/len(y_rel)
    return recall


def user_ap(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Average precision@k
    ----------------------------------
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: average precision metric for user recommendations
    """

    k_min = min(k, len(y_rec))
    ap = 0

    for i in range(k_min):
        if y_rec[i] in y_rel:
            p_i = len(set(y_rec[:i+1]).intersection(set(y_rel)))/(i+1)
            ap += p_i
    return ap/k
    

def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    NDCG@k
    ----------------------------------
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: ndcg metric for user recommendations
    """
    dcg = 0
    idcg = 0
    k_min = min(k, len(y_rec))
    for i in range(k_min):
        dcg += 1/np.log2(i+2) if y_rec[i] in y_rel else 0
        idcg += 1/np.log2(i+2)
    return dcg/idcg
    

def user_rr(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Reciprocal rank@k
    ----------------------------------
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: reciprocal rank for user recommendations
    """
    rr = 0
    for i in range(min(k, len(y_rec))):
        if y_rec[i] in y_rel:
            rr = 1/(i+1)
            break
    return rr
    