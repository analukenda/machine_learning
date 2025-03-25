import itertools as it

def rand_index_score(y_gold, y_predict):
    # Vaš kôd ovdje...
    a=0
    b=0
    N=len(y_gold)
    gold=it.combinations(y_gold,2)
    pred = it.combinations(y_predict,2)
    for gold_,pred_ in zip(gold,pred):
        if gold_[0]==gold_[1] and pred_[0]==pred_[1]:
            a+=1
        if gold_[0]!=gold_[1] and pred_[0]!=pred_[1]:
            b+=1
    return (a+b)/(N*(N-1)/2)