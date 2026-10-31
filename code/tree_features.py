"""Input-only features for numeric-tree verification of supplied candidates.

This module never reads data files or references. Arithmetic helpers inspect
bounded numeric ASTs only and cannot execute candidate-provided code.
"""
import math
import re
import numpy as np
from student_features import prepare_inputs
from equation_features import equation_stats

NUM = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")

def number(s):
    try:
        if "/" in s:
            a,b=s.split("/")
            return float(a.replace(",",""))/float(b.replace(",",""))
        return float(s.replace(",",""))
    except (ValueError,ZeroDivisionError):
        return None

def prepare_extra(bank, arrays):
    """Return official groups, embeddings, basic32 and75 extra diagnostics."""
    answers,groups,emb,basic=prepare_inputs(bank,arrays)
    rows=[]
    for row,row_answers,group in zip(bank,answers,groups):
        q=row["question"].lower()
        qnums=[number(s) for s in NUM.findall(q)]
        qnums=[v for v in qnums if v is not None]
        textsets=[set(re.findall(r"[a-z]+",c["text"].lower())) for c in row["candidates"]]
        avals=[number(a) if a is not None else None for a in row_answers]
        vals=[a for a in avals if a is not None]
        for k,(candidate,answer) in enumerate(zip(row["candidates"],row_answers)):
            text=candidate["text"];low=text.lower();a=avals[k]
            other_same=len(group.get(answer,[]))-1 if answer is not None else -1
            feats=equation_stats(text,answer)
            nums=[number(s) for s in NUM.findall(text)]
            nums=[v for v in nums if v is not None]
            if a is not None:
                alog=math.log1p(abs(a))
                repeat=sum(abs(v-a)<1e-5 for v in nums)
                freq_q=sum(abs(v-a)<1e-5 for v in qnums)
                aq=[math.log1p(abs(a-v)) for v in qnums]
                afinal=sum(abs(v-a)<1e-5 for v in nums[-5:])
            else:
                alog,repeat,freq_q,aq,afinal=0,0,0,[],0
            feats += [other_same,len(group),float(a is not None),alog,
                      float(a is not None and abs(a-round(a))<1e-5),
                      float(a is not None and a<0),float(a==0),float(a==1),
                      math.log1p(repeat),math.log1p(freq_q),min(aq) if aq else 0,
                      max(aq) if aq else 0,afinal,math.log1p(len(qnums)),math.log1p(len(q)),
                      float(a is not None and vals and a==max(vals)),
                      float(a is not None and vals and a==min(vals))]
            for word in ["not","instead","mistake","error","correct","assum","round",
                         "approximately","approx","remaining","left","total","per","each",
                         "percent","half","twice","more","less","need","cost","profit"]:
                feats.append(math.log1p(low.count(word)))
                feats.append(float(word in q)*math.log1p(low.count(word)))
            words=textsets[k]
            same=[len(words&textsets[j])/max(1,len(words|textsets[j]))
                  for j in range(4) if j!=k and row_answers[j]==answer]
            different=[len(words&textsets[j])/max(1,len(words|textsets[j]))
                       for j in range(4) if j!=k and row_answers[j]!=answer]
            feats += [max(same or [0]),np.mean(same) if same else 0,
                      max(different or [0]),np.mean(different) if different else 0]
            rows.append(feats)
    extra=np.asarray(rows,dtype=np.float32).reshape(len(bank),4,-1)
    return answers,groups,emb,np.concatenate([basic,extra],axis=-1)

def feature_array(emb,text,groups,mode):
    rel=np.zeros((len(groups),4,5),dtype=np.float32)
    for i,gs in enumerate(groups):
        for ids in gs.values():
            rel[i,ids,0]=len(ids)
            rel[i,ids,1]=len(gs)
            rel[i,ids,2]=max(map(len,gs.values()))
            rel[i,ids,3]=len(ids)==max(map(len,gs.values()))
            rel[i,ids,4]=1
    if mode=="raw":
        return np.concatenate([emb,text,rel],axis=2)
    if mode=="center":
        return np.concatenate([emb-emb.mean(1,keepdims=True),text,text-text.mean(1,keepdims=True),rel],axis=2)
    if mode=="both":
        return np.concatenate([emb,emb-emb.mean(1,keepdims=True),text,text-text.mean(1,keepdims=True),rel],axis=2)
    raise ValueError("Invalid tree feature mode")
