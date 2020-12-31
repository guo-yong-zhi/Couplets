import sys
from model import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device", DEVICE)
class VOCAB:
    pass

import json
with open('vocab.json') as f:
    itos_ = json.load(f)

stoi_ = {v:k for k,v in enumerate(itos_)}
VOCAB.itos = itos_
VOCAB.stoi = stoi_
n_vocab = len(VOCAB.itos)

import utils as ut

PAD_TOKEN = "#"
MASK_TOKEN = "[]"

class Batch:
    def __init__(self, src=None, trg=None, 
                 src_pad_mask=None, trg_pad_mask=None,
                 trg_mask=None, trg_y=None):
        self.src=src
        self.trg=trg
        self.src_pad_mask=src_pad_mask
        self.trg_pad_mask=trg_pad_mask
        self.trg_mask=trg_mask
        self.trg_y=trg_y
    def __repr__(self):
        return 'Batch{\n  src: %s\n  trg: %s}'%(itos(self.src.cpu().numpy()),
                                  itos(self.trg.cpu().numpy()))
    
def format_couplets_str(first, second='', stoi={}, mask_token="[]"):
    first = list(first.replace(' ', '').replace(',', '，').replace('.', '。'))
    second = [s.replace(' ', mask_token).replace('-', mask_token) for s in second]
    l = (len(first)-len(second))
    assert l >= 0
    second = second + [mask_token] * l
    return [stoi.get(s, s) for s in first], [stoi.get(s, s) for s in second]

def str_to_batch(first, second, stoi, mask_token):
    src, trg = format_couplets_str(first, second, stoi, mask_token)
    src = tc.LongTensor(src).unsqueeze(0)
    trg = tc.LongTensor(trg).unsqueeze(0)
    src_pad_mask = tc.ones_like(src, dtype=bool)
    trg_pad_mask = tc.ones_like(trg, dtype=bool)
    trg_mask = trg != stoi[mask_token]
    return Batch(src, trg, src_pad_mask,trg_pad_mask,trg_mask)

def itos(i, itos=VOCAB.itos):
    return ut.apply(''.join, ut.apply(itos.__getitem__, i), 
                    at=lambda c:not ut.iscollection(c[0][0]))

model = make_model(n_vocab, n_vocab).to(DEVICE)
# model.load_state_dict(torch.load('model_state_share.pt', map_location=DEVICE))
state_dict = torch.hub.load_state_dict_from_url('https://github.com/guo-yong-zhi/Couplets/releases/download/1.0/model_state_share.pt', model_dir='.', map_location=DEVICE)
model.load_state_dict(state_dict)

from decode_search import *

def match_couplet_onepass(first, second=''):
    '一次性输出，模型调用1次，效果不好'
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    out = model(batch.src.to(DEVICE), batch.trg.to(DEVICE), 
                batch.src_pad_mask.unsqueeze(1).to(DEVICE), batch.trg_pad_mask.unsqueeze(1).to(DEVICE))
    _, ind = torch.max(model.generator(out), dim = -1)
    return itos(ind.cpu().numpy())

def match_couplet_greedy(first, second=''):
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    trg_pred = decode_greedy(model, batch.src.to(DEVICE), batch.trg.to(DEVICE)
                             , mask_int=VOCAB.stoi[MASK_TOKEN], pad_int=VOCAB.stoi[PAD_TOKEN])
    return itos(trg_pred.cpu().numpy())

def match_couplet_beam_1D(first, second='', beamsize=5):
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    results = beam_decode_engine(model, batch.src.to(DEVICE), batch.trg.to(DEVICE), beamsize, decode_1D_step
                             , mask_int=VOCAB.stoi[MASK_TOKEN], pad_int=VOCAB.stoi[PAD_TOKEN])
    return itos([r.cpu().numpy() for r in results])

def match_couplet_beam_margin(first, second='', beamsize=5):
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    results = beam_decode_engine(model, batch.src.to(DEVICE), batch.trg.to(DEVICE), beamsize, decode_margin_step
                             , mask_int=VOCAB.stoi[MASK_TOKEN], pad_int=VOCAB.stoi[PAD_TOKEN])
    return itos([r.cpu().numpy() for r in results])

def match_couplet_beam_2D(first, second='', beamsize=5):
    '条件概率版本'
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    results = beam_decode_engine(model, batch.src.to(DEVICE), batch.trg.to(DEVICE), beamsize, decode_2D_step
                             , mask_int=VOCAB.stoi[MASK_TOKEN], pad_int=VOCAB.stoi[PAD_TOKEN])
    return itos([r.cpu().numpy() for r in results])

def match_couplet_beam_2D_2(first, second='', beamsize=5):
    '联合分布概率版本'
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    results = decode_beam_2D_2(model, batch.src.to(DEVICE), batch.trg.to(DEVICE), beamsize
                             , mask_int=VOCAB.stoi[MASK_TOKEN], pad_int=VOCAB.stoi[PAD_TOKEN])
    return itos([r.cpu().numpy() for r in results])

def decode_margin_step2(model, src, trg, mask, padmask, beamsize):
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
        r[0, max_pos] = int(i)
        result.append(r)
#     print(itos([r.cpu().numpy() for r in result]))
    return result

def beam_decode_engine2(model, src, trg, beamsize, decodestep, mask_int, pad_int):
    '候选的概率由logp_of_trg确定（联合分布）'
    padmask = (src!=pad_int).squeeze()
    result = [trg]
    while True:
        candidates = {}
        for trg in result:
            mask = (trg!=mask_int).squeeze()
            if mask.all(): 
                return result
            cands = decodestep(model, src, trg, mask, padmask, beamsize)
            for c in cands:
                candidates[tuple(c[0].tolist())] = c #去重
        cands = list(candidates.values())
        logps = [logp_of_trg(model, src, trgi, (trgi!=mask_int).squeeze(0), padmask) for trgi in cands]
        maxinds = max_n_1D(np.array(logps), min(len(logps), beamsize))
        result = [cands[i] for i in maxinds]
#         print(itos([r.cpu().numpy() for r in result]))
    return result

def match_couplet_beam_margin2(first, second='', beamsize=5):
    batch = str_to_batch(first, second, VOCAB.stoi, MASK_TOKEN)
    model.eval()
    results = beam_decode_engine2(model, batch.src.to(DEVICE), batch.trg.to(DEVICE), beamsize, decode_margin_step2
                             , mask_int=VOCAB.stoi[MASK_TOKEN], pad_int=VOCAB.stoi[PAD_TOKEN])
    return itos([r.cpu().numpy() for r in results])

array=list
def print_match_all(first, second='', beamsize=5, file=sys.stdout):
    print("上联:", first, file=file)
    print("onepass(not good):", match_couplet_onepass(first, second), file=file)
    print("greedy :", match_couplet_greedy(first, second), file=file)
    print("beam_1D:\n", array(match_couplet_beam_1D(first, second, beamsize)), file=file)
    print("beam_2D:\n", array(match_couplet_beam_2D(first, second, beamsize)), file=file)
    print("beam_2D_2:\n", array(match_couplet_beam_2D_2(first, second, beamsize)), file=file)
    print("beam_margin:\n", array(match_couplet_beam_margin(first, second, beamsize)), file=file)
    print("beam_margin2:\n", array(match_couplet_beam_margin2(first, second, beamsize)), file=file)
    
print('输入上下联，或`q`退出')
print('如有下联用`|`隔开，下联空字用空格或减号占位')
print('输入例：\n白日依山尽\n白日依山尽|-河-海\n白日依山尽|明月\n')
while True:
    print("请输入：", end='')
    i = input().split('|')
    first = i[0]
    second = i[1] if len(i)>1 else ''
    if first.startswith('q'): break
    print_match_all(first, second)
    print('='*8)
