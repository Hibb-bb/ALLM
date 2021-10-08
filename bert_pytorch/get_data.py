from datasets import load_dataset
from bpemb import BPEmb


def sent_split(sents):
    res=[]
    for sent in sents:
        sent = sent.split('.')
        for s in sent:
            res.append(s.replace('\n', ''))
    print('sents size', len(res))
    return res
    
def merge_sents(sents):
    for i in range(len(sents)-1):
        sents[i] = sents[i] + ['\t'] + sents[i+1]
        sents[i] = ' '.join(sents[i])
    sents[-1] = sents[-2]
    # print(sents[-1])
    # sents[-1] = ' '.join(sents[-1])
    return sents

train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')

bpemb = BPEmb(lang="en", vs=25000)

train_text = [t['text'] for t in train]
train_text = sent_split([t for t in train_text if len(t.replace('\n', '')) > 0])
print(train_text[:5])

test_text = [t['text'] for t in test]
test_text = sent_split([t for t in test_text if len(t.replace('\n', '')) > 0])
print(test_text[:5])

train_text = merge_sents([bpemb.encode(t) for t in train_text])
print('bpe + merge')
print(train_text[:5])

test_text = merge_sents([bpemb.encode(t) for t in test_text])

with open('wikitext.train.txt', 'w') as f:
    for t in train_text:
        f.write(f'{t}\n')


with open('wikitext.test.txt', 'w') as f:
    for t in test_text:
        f.write(f'{t}\n')

# train_text= [t.insert(3, '\t') for t in train_text]
# test_text = [t.insert(3,'\t') for t in test_text]


