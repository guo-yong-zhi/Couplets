import torch
import torch as tc
import numpy as np
# greedy ============
def decode_greedy_step(model, src, trg, mask, padmask):
    model.eval()
#     print(padmask.unsqueeze(0).unsqueeze(1))
    out = model(src, trg, padmask.unsqueeze(0).unsqueeze(1), padmask.unsqueeze(0).unsqueeze(1))
    s, ind = torch.max(model.generator(out), dim = -1)
    s[0][mask] = -1e9
    s[0][~padmask] = -1e9
    _, pos = torch.max(s, dim = -1)
    ind_i = ind[0][pos].item()
    return pos.item(), ind_i

def decode_greedy(model, src, trg, mask_int, pad_int):
    '贪心解码，一次填一个字，model最多调用len(trg)次'
    while True:
        mask = (trg!=mask_int).squeeze()
        if mask.all(): break
        padmask = (src!=pad_int).squeeze()
        p, i = decode_greedy_step(model, src, trg, mask, padmask)
        trg[0][p]=i
#         print(p, i, VOCAB.itos[i])
    return trg

# 1D beam ============
def decode_1D_step(model, src, trg, mask, padmask, beamsize):
    '''
    各候选位置选最可能的字
    padmask为True的是有效字，mask为True的是句中的给定字（非mask）
    '''
    model.eval()
#     print(padmask.unsqueeze(0).unsqueeze(1))
    assert src.size(0) == 1
    out = model(src, trg, padmask.unsqueeze(0).unsqueeze(1), padmask.unsqueeze(0).unsqueeze(1))
    g = model.generator(out)
    s, ind = torch.max(g, dim = -1)
    ignore = mask | (~padmask)
    s[0][ignore] = -1e9
    snp = s[0].detach().cpu().numpy() 
    pos = max_n_1D(np.array(snp), min(len(snp), beamsize))
    result = []
    keep_pos = pos[:(~ignore).sum().item()]
    for p in keep_pos:
        r = tc.clone(trg)
        r[0, p] = ind[0][p].item()
        result.append(r)
    logps = snp[keep_pos]
#     print(itos([r.cpu().numpy() for r in result]))
    return result, logps

def decode_margin_step(model, src, trg, mask, padmask, beamsize):
    '''
    各候选位置选最可能的字 + 最可能的位置选候选字，伪二维搜索
    padmask为True的是有效字，mask为True的是句中的给定字（非mask）
    '''
    model.eval()
#     print(padmask.unsqueeze(0).unsqueeze(1))
    assert src.size(0) == 1
    out = model(src, trg, padmask.unsqueeze(0).unsqueeze(1), padmask.unsqueeze(0).unsqueeze(1))
    g = model.generator(out)
    s, ind = torch.max(g, dim = -1)
    ignore = mask | (~padmask)
    s[0][ignore] = -1e9
    snp = s[0].detach().cpu().numpy() 
    pos = max_n_1D(np.array(snp), min(len(snp), beamsize//2))
    result = []
    keep_pos = pos[:(~ignore).sum().item()]
    for p in keep_pos:
        r = tc.clone(trg)
        r[0, p] = ind[0][p].item()
        result.append(r)
    logps = snp[keep_pos]
    max_pos = pos[0]
    g = g[0, max_pos].detach().cpu().numpy()
    ind = max_n_1D(g, min(len(g), beamsize-len(result)))
    
    for i in ind:
        r = tc.clone(trg)
        r[0, max_pos] = i
        result.append(r)
#     print(itos([r.cpu().numpy() for r in result]))
    return result, np.concatenate((logps, g[ind]))

def beam_decode_engine(model, src, trg, beamsize, decodestep, mask_int, pad_int):
    '按条件概率的方式决定候选的概率'
    padmask = (src!=pad_int).squeeze()
    result = [trg]
    resultlogp = [0]
    while True:
        candidates = {}
        for trg, rlp in zip(result, resultlogp):
            mask = (trg!=mask_int).squeeze()
            if mask.all(): 
                return result
            cands, logpi = decodestep(model, src, trg, mask, padmask, beamsize)
            for c,li in zip(cands, logpi):
                candidates[tuple(c[0].tolist())] = c, rlp+li #去重
        cands = [c[0] for c in candidates.values()]
        logps = [c[1] for c in candidates.values()]
        maxinds = max_n_1D(np.array(logps), min(beamsize, len(logps)))
        result = [cands[i] for i in maxinds]
        resultlogp = [logps[i] for i in maxinds]
#         print(candidates)
#         print(itos([r.cpu().numpy() for r in result]))
    return result


# 2D beam ============
def max_n_2D(arr, n):
    arr = -arr
    flat_indices = np.argpartition(arr.ravel(), n-1)[:n]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    min_elements = arr[row_indices, col_indices]
    min_elements_order = np.argsort(min_elements)
    row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]
    return row_indices, col_indices

def max_n_1D(arr, n):
    arr = -arr
    indices = np.argpartition(arr, n-1)[:n]
    min_elements = arr[indices]
    min_elements_order = np.argsort(min_elements)
    ordered_indices = indices[min_elements_order]
    return ordered_indices

def decode_2D_step(model, src, trg, mask, padmask, beamsize):
    '''
    各位置各字选候选，二维，返回当前候选的概率
    padmask为True的是有效字，mask为True的是句中的给定字（非mask）
    '''
    model.eval()
    out = model(src, trg, padmask.unsqueeze(0).unsqueeze(1), padmask.unsqueeze(0).unsqueeze(1))
    assert out.size(0) == 1
    g = model.generator(out).squeeze(0)
    g[mask] = -1e9
    g[~padmask] = -1e9
    gnp = g.detach().cpu().numpy()   
    rind, cind = max_n_2D(gnp, beamsize)
    result = []
    for i,v in zip(rind, cind):
        r = tc.clone(trg)
        r[0, i] = v
        result.append(r)
    return result, gnp[rind, cind]


def decode_2D_step_2(model, src, trg, mask, padmask, beamsize):
    '''
    各位置各字选候选，二维，不返回当前候选的概率
    padmask为True的是有效字，mask为True的是句中的给定字（非mask）
    '''
    model.eval()
    out = model(src, trg, padmask.unsqueeze(0).unsqueeze(1), padmask.unsqueeze(0).unsqueeze(1))
    assert out.size(0) == 1
    g = model.generator(out).squeeze(0)
    g[mask] = -1e9
    g[~padmask] = -1e9
    gnp = g.detach().cpu().numpy()   
    rind, cind = max_n_2D(gnp, beamsize)
    result = []
    for i,v in zip(rind, cind):
        r = tc.clone(trg)
        r[0, i] = v
        result.append(r)
    return result

def logp_of_trg(model, src, trg, mask, padmask):
    '计算无mask部分的对数概率和'
    model.eval()
    out = model(src, trg, padmask.unsqueeze(0).unsqueeze(1), padmask.unsqueeze(0).unsqueeze(1))
    assert out.size(0) == 1
    g = model.generator(out).squeeze(0)
    inds = trg[0][mask&padmask]
    loss_all = g[mask&padmask,inds].sum().item()
    return loss_all

def decode_beam_2D_2(model, src, trg, beamsize, mask_int, pad_int):
    '候选的概率由logp_of_trg确定（联合分布）'
    padmask = (src!=pad_int).squeeze()
    result = [trg]
    while True:
        candidates = {}
        for trg in result:
            mask = (trg!=mask_int).squeeze()
            if mask.all(): 
                return result
            cands = decode_2D_step_2(model, src, trg, mask, padmask, beamsize)
            for c in cands:
                candidates[tuple(c[0].tolist())] = c #去重
        cands = list(candidates.values())
        logps = [logp_of_trg(model, src, trgi, (trgi!=mask_int).squeeze(0), padmask) for trgi in cands]
        maxinds = max_n_1D(np.array(logps), beamsize)
        result = [cands[i] for i in maxinds]
#         print(itos([r.cpu().numpy() for r in result]))
    return result

