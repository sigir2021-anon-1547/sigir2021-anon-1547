# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from __future__ import print_function
import numpy as np

import cython

cimport numpy as np

import random
from collections import defaultdict

from tqdm import tqdm

"""
Compiler directives:
https://github.com/cython/cython/wiki/enhancements-compilerdirectives
Cython tutorial:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
Credit to https://github.com/luzai
"""

# Main interface
cpdef evaluate_cy(indices, q_pids, g_pids, q_camids, g_camids, max_rank):
    indices = np.asarray(indices, dtype=np.int64)
    q_pids = np.asarray(q_pids, dtype=np.int64)
    g_pids = np.asarray(g_pids, dtype=np.int64)
    q_camids = np.asarray(q_camids, dtype=np.int64)
    g_camids = np.asarray(g_camids, dtype=np.int64)
    return eval_func_cy(indices, q_pids, g_pids, q_camids, g_camids, max_rank)


cdef long[:] k_list = np.array([1,5,10,20,50])


cdef top_k_retrieval(float[:] row_matches, long[:] k):
    cdef long[:] results = np.zeros((k.shape[0]), dtype=np.int64)
    cdef long kk
    cdef long tmp

    for k_idx in range(k.shape[0]):
        kk = k[k_idx]
        tmp = int(np.any(row_matches[:kk]))
        results[k_idx] = tmp
    return results


cdef eval_func_cy(long[:,:] indices, long[:] q_pids, long[:] g_pids,
                  long[:] q_camids, long[:] g_camids, long max_rank=50):

    cdef long num_q = indices.shape[0]
    cdef long num_g = indices.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    
    cdef:
        # long[:,:] indices = np.argsort(distmat, axis=1)
        long[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0. # number of valid query

        long q_idx, q_pid, q_camid, g_idx
        long[:] order = np.zeros(num_g, dtype=np.int64)
        long keep

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] cmc = np.zeros(num_g, dtype=np.float32)
        long num_g_real, rank_idx
        unsigned long meet_condition

        float num_rel
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum

        long num_top_k = k_list.shape[0]
        long[:,:] topk_results = np.zeros((num_q, num_top_k), dtype=np.int64)
        long[:] top_k_res
        float[:,:] single_performance = np.zeros((num_q, 3), dtype=np.float32)

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0
        
        for g_idx in range(num_g):
            if (g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid):
                raw_cmc[num_g_real] = matches[q_idx][g_idx]
                num_g_real += 1
                if matches[q_idx][g_idx] > 1e-31:
                    meet_condition = 1
        
        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            continue

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        AP = tmp_cmc_sum / num_rel
        all_AP[q_idx] = AP

        # Save AP for each query to allow finding worst performing samples
        single_performance[q_idx, 0] = q_idx
        single_performance[q_idx, 1] = q_pid
        single_performance[q_idx, 2] = AP

        # Get topk accuracy for topk
        top_k_res = top_k_retrieval(tmp_cmc, k_list)
        for k_idx in range(num_top_k):
            topk_results[q_idx, k_idx] = top_k_res[k_idx]

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q
    
    cdef float mAP = 0
    for q_idx in range(num_q):
        mAP += all_AP[q_idx]
    mAP /= num_valid_q

    # compute average topk
    cdef float[:] avg_topk = np.zeros(len(k_list), dtype=np.float32)
    for topk_idx in range(len(k_list)):
        for q_idx in range(num_q):
            avg_topk[topk_idx] += topk_results[q_idx, topk_idx]
        avg_topk[topk_idx] /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), mAP, np.asarray(avg_topk).astype(np.float32), np.asarray(single_performance).astype(np.float32)


# Compute the cumulative sum
cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, long n):
    cdef long i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]