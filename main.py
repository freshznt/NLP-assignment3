# 导入必要的库
import os
import gensim
import matplotlib.font_manager
from gensim.models import Word2Vec
import jieba
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# 步骤1：数据预处理
def load_data(file_path,stopWordsPath):
    stopWords = []
    with open(stopWordsPath, 'r', encoding='utf-8') as stopWordFile:
        stopWords.extend([line.strip() for line in stopWordFile.readlines()])
    text = []
    for file in os.listdir(file_path):
        with open(os.path.join(file_path,file), 'r', encoding='utf-8') as file:
            for line in file:
                line = line.replace('----〖新语丝电子文库(www.xys.org)〗', '')
                line = line.replace('本书来自www.cr173.com免费txt小说下载站', '')
                line = line.replace('更多更新免费电子书请关注www.cr173.com', '')
                words = jieba.lcut(line)
                words = [word for word in words if word.isalnum() and word not in stopWords]
                if len(words) != 0:
                    text.append(words)
    return text



# 步骤2：训练Word2Vec模型
def train_word2vec_model(sentences, sg=0, vector_size=20, window=5, min_count=5, epochs=7,negative=10):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,negative=negative, sg=sg)
    return model


# 步骤3：计算词向量的语义距离
def compute_similarity(model, word1, word2, mode='cos'):
    if mode == 'cos':
        return model.wv.similarity(word1, word2)
    elif mode == 'euclidean':
        vec1 = model.wv.get_vector(word1)
        vec2 = model.wv.get_vector(word2)
        return np.linalg.norm(vec1-vec2)

# 步骤4：词语的聚类分析
def cluster_words(model,wordlists, num_clusters=3):
    vectors = []
    for word in wordlists:
        vectors.append(list(model.wv.get_vector(word)))
    word_vectors = np.array(vectors,dtype=float)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)
    return word_vectors,kmeans


def visualize_clusters(word,word_vectors, kmeans):
    # tsne = TSNE(n_components=2, random_state=0)
    zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf',size=12)
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 10))
    for i, _ in enumerate(reduced_vectors):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], c='C' + str(kmeans.labels_[i]))
        plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1],word[i],fontproperties=zhfont1,alpha=1,color='black')
    plt.show()


# 步骤5：验证词向量的有效性
def find_similar_words(model, word, topn=20):
    return model.wv.most_similar(word, topn=topn)


# 主程序
if __name__ == "__main__":
    # 读取并预处理数据
    file_path = './data/corpus_utf8/'  # 请更改为实际文件路径
    stopWordsPath = './data/stopwords/cn_stopwords.txt'
    invest_words = ['郭靖','黄蓉','杨过','赵敏','张无忌','张翠山','郭襄','小龙女','武三通','周芷若',
                    '一阳指','六脉','降龙十八掌','打狗棒法','乾坤','太极拳','神剑','蛤蟆功',
                    '衢州','天津','汴梁','信阳','广州','广西','凉州','白马寺','镇江','太原']
    text = load_data(file_path,stopWordsPath)

    # 训练Word2Vec模型
    model = train_word2vec_model(text,sg=1)

    # 计算词语相似度
    similarity = compute_similarity(model, '郭靖', '黄蓉')
    print(f"郭靖 和 黄蓉 的相似度: {similarity}")

    # 验证词向量有效性
    similar_words = find_similar_words(model, '郭靖')
    print(f"与 郭靖 最相似的词语: {similar_words}")

    # 聚类分析
    word_vectors,kmeans = cluster_words(model,invest_words)
    visualize_clusters(invest_words,word_vectors,kmeans)



    paragraph1 = "郭靖和黄蓉在一起打怪升级。"
    # paragraph2 = "乔峰和阿朱在大漠中行侠仗义。"
    # paragraph_similarity = compute_paragraph_similarity(model, paragraph1, paragraph2)
    # print(f"两个段落的相似度: {paragraph_similarity}")
