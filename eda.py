import json
import re

import jieba
import synonyms
import random
from random import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm



# 停用词列表，默认使用哈工大停用词表
f = open('hit_stopwords.txt', encoding='utf-8')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])

# 考虑到与英文的不同，暂时搁置
# 文本清理
'''
import re
def get_only_chars(line):
    #1.清除所有的数字
'''


########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words and word not in add_word]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    return synonyms.nearby(word)[0]







########################################################################
# EDA函数
def eda(sentence, alpha_sr=0.1, num_aug=9):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))

    # print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))


    # print(augmented_sentences)
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(seg_list)

    return augmented_sentences

##
# 测试用例
# 示例句子和实体列表
# sentence = "段继东，男，汉族，"
# entity = [
#     {"index": [0, 1, 2], "type": "NAME"},
#     {"index": [6, 7], "type": "RACE"}
# ]







if __name__ == '__main__':
    random.seed(2024)
    with open('weibo/train_aug.json', encoding='utf-8') as f:
        a, b = [], []

        for line in tqdm(f, desc="aug data"):
            l = json.loads(line)
            s = ''
            for text in l["text"]:
                s += text
            add_word = []  # 放在这里指挥保留当前句子的实体信息并用在下面进行mask
            ty = {}
            for e in l['labels']:
                start = e['index'][0]
                end = e['index'][-1] + 1
                fl = e['type']
                add_word.append(s[start:end])  # 获得该句子所有的实体信息
                ty[s[start:end]] = fl   # 获得该句子所有的类型信息
            # print(add_word)
            # for word in add_word:     #遍历所有的实体使其在在分词时把实体当作一个完整的词
            #     jieba.add_word(word)        #向jieba的分词库中加入实体使其保留实体进行分词。但是这种方法在遇到特殊字符时会出错。
            #还是应该使用MASK来MASK掉实体信息
            sim_sent = s
            # print(sim_sent)
            #对于add_word进行排序
            sorted_list = sorted(add_word, key=len, reverse=True)
            #然后对于句子中的实体，我们进行MASK
            for i,e in enumerate(sorted_list):
                s = s.replace(e,f"MA{i}SK")
                #同时我们把MA{i}SK加入结巴词典中
                jieba.add_word(f"MA{i}SK")



            augmented_sentences = eda(sentence=s)  #数据增强
            augmented_sentence = []
            #对于增强之后的句子进行恢复
            # 然后对于句子中的实体，我们进行MASK
            for sent in augmented_sentences:
                for i, e in enumerate(sorted_list):
                    sent = sent.replace(f"MA{i}SK", e)
                augmented_sentence.append(sent)



            augmented_sentences = list(set(augmented_sentence))    #去重
            aug_sent = []
            # 对于增强之后的句子我门选择一个余弦相似度最近的一个当作增强
            if len(augmented_sentences) == 1:
                augmented_sentences = augmented_sentences[0].replace(" ","")
                a.append([i for i in augmented_sentences])
                b.append(l['labels'])
            else:
                # 这里就灭有必要去判断句子是否是原句子了
                augmented = []
                for se in augmented_sentences.copy():
                    se = se.replace(" ", "")  # 替换空格
                    if se != sim_sent:
                        augmented.append(se)  # 去掉重复的，去掉和原句子相同的，就是最后我们增强之后的句子


                original_sentence = sim_sent
                # 使用 TF-IDF 向量化
                vectorizer = TfidfVectorizer().fit([original_sentence] + augmented)
                original_vector = vectorizer.transform([original_sentence])
                augmented_vectors = vectorizer.transform(augmented)

                # 计算相似度
                similarities = cosine_similarity(original_vector, augmented_vectors).flatten()

                # 找到最相似的句子
                most_similar_index = similarities.argmax()
                sim_sentence = augmented[most_similar_index]  # 在进行选择同义词时具有随机性
                # aug_sent.append()  # 最相似的句子
                a.append([i for i in sim_sentence])  # 存放所有的最相似的句子

                # sim_sentence = most_similar_sentence.replace(" ", "")

                # print("最相似的句子:", most_similar_sentence.replace(" ", ""))

                # 用于存放实体的所有下标
                entity_positions = {}
                # 用于存放已匹配的字符范围，防止重复匹配
                matched_ranges = []

                # 按照实体长度降序排列，确保较长的实体先匹配
                add_word.sort(key=lambda x: -len(x))  # add_word是为了保证我们的分词的结果把所有的实体存放在此列表中

                # 遍历实体信息，查找每个实体的位置
                for entity in add_word:
                    matches = list(re.finditer(re.escape(entity), ''.join(sim_sentence)))  # 合并分词后的文本
                    positions = []

                    # 遍历每个匹配项
                    for match in matches:
                        start, end = match.start(), match.end()

                        # 检查当前匹配是否与已匹配的范围重叠
                        if not any(start < r[1] and end > r[0] for r in matched_ranges):
                            positions.append((start, end))
                            matched_ranges.append((start, end))  # 记录这个范围

                    # 如果匹配到的位置，添加到实体的下标列表
                    if positions:
                        entity_positions[entity] = positions

                all_label = []
                # 输出实体对应的所有下标
                for entity, positions in entity_positions.items():

                    # print("增强之后于原始句子最相似的一个句子："+sim_sentence)
                    for position in positions:
                        # 每次都需要创建
                        dic = {}
                        start = position[0]
                        end = position[1]
                        index = [j for j in range(start, end)]
                        dic['index'] = index
                        dic["type"] = ty[entity]
                        all_label.append(dic)
                b.append(all_label)


    train_data = [{'text': text, 'labels': entities} for text, entities in zip(a, b)]
    with open('weibo/train_aug_already.json', "w", encoding="utf-8") as fw:  # 打开指定文件
        for line in train_data:
            json.dump(line, fw, ensure_ascii=False)
            fw.write('\n')
