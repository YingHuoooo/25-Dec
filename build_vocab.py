import json

def build_skeleton_vocab(data_file):
    # 1. 定义特殊 Token
    # [PAD]: 填充, [BOS]: 开始, [EOS]: 结束, [ARG]: 参数槽位
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[ARG]": 3}
    idx = 4
    
    # 2. 扫描所有数据
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            skeleton = data['skeleton']
            
            # 简单的分词：把 'result', '=', 'cq.Workplane', '(', ')', '.', 'box' 分开
            # 这里做一个简化的正则分词逻辑，或者直接按字符切分太细
            # 建议：将 skeleton 字符串预处理成 token 列表
            tokens = tokenize_skeleton(skeleton)
            
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
    
    print(f"词表构建完成，大小: {len(vocab)}")
    return vocab

def tokenize_skeleton(skel_str):
    """
    简单的分词器：把代码字符串切分成 Token
    例子: "cq.Workplane().box([ARG])" -> ['cq.Workplane', '(', ')', '.', 'box', '(', '[ARG]', ')']
    """
    # 这是一个简单的替换技巧，给符号加空格，然后 split
    for char in ['(', ')', '.', ',', '=']:
        skel_str = skel_str.replace(char, f" {char} ")
    return skel_str.split()

if __name__ == "__main__":
    vocab = build_skeleton_vocab("complex_cnc_dataset_qwen2.jsonl")
    with open("skeleton_vocab.json", "w") as f:
        json.dump(vocab, f)