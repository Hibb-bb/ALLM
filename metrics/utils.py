# -*- coding: utf-8 -*-

def tokenize(text, max_len=128):
    return text[:max_len]


def load_tsv(data_file, max_seq_len, s1_idx=0, s2_idx=1, targ_idx=2, idx_idx=None,
             targ_map=None, targ_fn=None, skip_rows=0, delimiter='\t'):
    '''Load a tsv '''
    sent1s, sent2s, targs, idxs = [], [], [], []
    with open(data_file, 'r', encoding='utf-8') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.strip().split(delimiter)
                sent1 = tokenize(row[s1_idx], max_seq_len)
                if (targ_idx is not None and not row[targ_idx]) or not len(sent1):
                    continue

                if targ_idx is not None:
                    if targ_map is not None:
                        targ = targ_map[row[targ_idx]]
                    elif targ_fn is not None:
                        targ = targ_fn(row[targ_idx])
                    else:
                        targ = int(row[targ_idx])
                else:
                    targ = 0

                if s2_idx is not None:
                    sent2 = tokenize(row[s2_idx], max_seq_len)
                    if not len(sent2):
                        continue
                    sent2s.append(sent2)

                if idx_idx is not None:
                    idx = int(row[idx_idx])
                    idxs.append(idx)

                sent1s.append(sent1)
                targs.append(targ)

            except Exception as e:
                print(e, f' file: data_file, row: row_idx')
                continue

    if idx_idx is not None:
        return sent1s, sent2s, targs, idxs
    else:
        return sent1s, sent2s, targs
