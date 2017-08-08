# -*- coding: utf-8 -*-
#!/usr/bin/python
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.preprocessing as sp
from sklearn.grid_search import GridSearchCV
import jieba
import jieba.analyse
from gensim.models import word2vec
import codecs
import pandas as pd
import xgboost as xgb

#  学习参数
params = {
    "objective": "binary:logistic",
    "eta": 0.25,
    'max_depth': 9,
    'silent': 1,
    'nthread': 4,
    'n_estimators': 100,
}


# 预处理文件
def pre_process(file_name, columns=['user_id', 'content']):
    print "loading data from", file_name, '...'
    df = pd.read_csv(
        file_name,
        names=columns,
        sep='\t',
        dtype={'content': np.str,
               'user_id': np.str})
    tt = {}
    for i in range(df.shape[0]):
        # print "read", i
        user_id, content = df['user_id'][i], df["content"][i]
        if type(content) is np.float or type(user_id) is np.float:
            continue
        # print user_id, content
        if tt.has_key(user_id):
            tt[user_id] = '。'.join([tt[user_id], content])
        else:
            tt[user_id] = content
    print "total user:", len(tt.keys())
    return tt.items()


# 分词
def process_file(contents, result_file_name):
    jieba.enable_parallel(4)
    # with open(file_name, 'r') as f:
    #     sentence = "\n".join(f.readlines())
    #     print ' '.join(jieba.analyse.extract_tags(sentence))

    with codecs.open(result_file_name, 'w', "utf-8") as f2:
        cnt = 0
        for uid, line in contents:
            if line is None or line.__len__() == 0:
                continue
            line = line.replace(" ", "")
            #print ' '.join(jieba.lcut(line, HMM=False))
            if cnt % 1000 == 0:
                print "line ", cnt
            f2.write(" ".join([
                x for x in jieba.lcut(line, HMM=False)
                if check_contain_chinese(x) and len(str(x)) > 4
            ]))
            f2.write('\n')
            cnt += 1


# TF-IDF 提取特征项
def tf_idf(file_name):
    corpus = []
    with open(file_name) as f:
        corpus = f.readlines()
    vectorizer = CountVectorizer()  #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  #该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(
        corpus))  #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  #获取词袋模型中的所有词语
    weight = tfidf.toarray()  #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    top_ks = []
    for i in range(
            len(weight)):  #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重

        if i % 100 == 0:
            print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"

        top_k = sorted(
            [(word[j], weight[i][j]) for j in range(len(word))],
            cmp=lambda x, y: int((y[1] - x[1]) * 10**6))[:20]
        top_ks.append(top_k)
        if i % 100 == 0:
            print '\n'.join(map(lambda x: x[0] + " " + str(x[1]), top_k))

        if i > 10000:
            break

    return top_ks


# 创建词向量
def create_word2vec_model(fenci_file, model_file):
    if not os.path.isfile(model_file):
        sentences = word2vec.LineSentence(fenci_file)
        model = word2vec.Word2Vec(sentences, size=200, min_count=0)
        model.save(model_file)
        return model
    else:
        return word2vec.Word2Vec.load(model_file)


# 预处理生成唯一点
def pre_create_word_vector(mdl, top_ks):
    print "pre create vecort ..."
    word_dict = {}
    wgts = []
    for top_k in top_ks:
        sum = 0.00001
        wrds = []
        for wd, wht in top_k:
            if not word_dict.has_key(wd):
                word_dict[wd] = mdl.wv[wd]
            wrds.append([wd, wht])
            sum += wht

        for wrd in wrds:
            wrd[-1] /= sum
        wgts.append(wrds)

    return word_dict.keys(), word_dict.values(), wgts


# 计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# 随机生成初始的质心（ng的课说的初始方式是随机选K个点）    
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(np.array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


# k-means 算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,
                                      2)))  #create mat to assign data points 
    #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(
                m):  #for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print centroids
        for cent in range(k):  #recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[
                0]]  #get all the point in this cluster
            centroids[cent, :] = np.mean(
                ptsInClust, axis=0)  #assign centroid to mean 
    return centroids, clusterAssment


# 保存数据
def save_processed_data(uids, weightset, words_keys, cluster_assment,
                        feature_num, flags):
    data = {"uid": uids, 'izombie': flags[:len(uids)]}

    clu_map = {}
    for i, wd in enumerate(words_keys):
        clu_map[wd] = cluster_assment[i, 0]
        if not data.has_key("fe" + str(cluster_assment[i, 0])):
            data["fe" + str(cluster_assment[i, 0])] = [0] * len(uids)

    for i in range(len(uids)):
        for wrd in weightset[i]:
            if clu_map.has_key(wrd[0]):
                data["fe" + str(clu_map[wrd[0]])][i] += int(
                    round(wrd[1] * feature_num))
    data_csv = pd.DataFrame(data)
    data_csv.to_csv("./data/social/data.csv", index=False)


# 保存各个词在所属聚类
def save_cluster_words(cluster_assment, words_keys, file_name):
    with open(file_name, 'w') as f:
        for i, wd in enumerate(words_keys):
            f.write(str(wd) + " " + str(cluster_assment[i, 0]) + '\n')


# xgboost 分类算法
def run_xgboost(train_file):
    print "loading train data from", train_file, '...'
    df = pd.read_csv(train_file, header=0)
    dx = df.iloc[:, :-2].values
    dy = df.iloc[:, -2].values
    print dx
    print dy
    xg_train = xgb.DMatrix(dx, label=dy)
    print "begin train..."
    bst = xgb.cv(params, xg_train, 100, metrics={'error'}, seed=23)
    print "train end\nsaving..."
    with open('./data/social/gb.txt', 'w') as f:
        f.write(bst.to_string())


# 调参数
def tune_parameters(train_file):
    print "load data ..."
    dataset = pd.read_csv(train_file, header=0)
    train_X = dataset.iloc[:, :-2].values
    train_Y = dataset.iloc[:, -2].values
    xg_train = xgb.DMatrix(train_X, label=train_Y)

    param_grid = {
        'learning_rate':
        [0.1, 0.13, 0.15, 0.17, 0.2, 0.22, 0.25, 0.27, 0.3, 0.32, 0.35, 0.4],
        "max_depth": [7, 8, 9, 10, 11, 12],
        "n_estimators": [70, 80, 90, 100, 110, 120],
    }
    print "create classifier..."
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=10,
        silent=True,
        objective="binary:logistic",
        n_jobs=10,
        random_state=32)
    searcher = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
    #train_Y = [sum(x) for x in train_Y]
    #train_Y = sp.label_binarize(train_Y, classes=range(0, 10))
    #print train_Y.shape, train_X.shape
    #print train_Y[66, 9]
    print "fitting ..."
    searcher.fit(train_X, train_Y)
    print searcher.grid_scores_
    print '>>' * 10
    print searcher.best_params_
    print '<<' * 10
    print searcher.best_score_


# 检查是否为中文字符
def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            continue
        else:
            return False
    return True


if __name__ == '__main__':
    print "begin process..."
    contents = pre_process('data/social/fake_account.csv')
    contents.extend(
        pre_process(
            'data/social/legitimate_account.csv',
            columns=[
                'user_id', 'postdate', 'retweet_count', 'comment_count',
                'like_count', 'content'
            ]))
    process_file(contents, './data/social/fenci.txt')

    top_ks = tf_idf('./data/social/fenci.txt')
    mdl = create_word2vec_model('./data/social/fenci.txt',
                                './model/fake_account.fc')

    words_keys, words_vals, weightset = pre_create_word_vector(mdl, top_ks)

    centroids, clusterAssment = kMeans(
        np.array(words_vals, dtype=np.float), 20)
    # print centroids
    print clusterAssment[:10, :]  # 前10个 所属cluster以及距离

    save_cluster_words(clusterAssment, words_keys, './data/social/cw.txt')

    flags = ([0] * 507)  # fake user
    flags.extend([1] * 10160)  # legal user
    save_processed_data(['c' + str(i) for i, _ in enumerate(top_ks)],
                        weightset, words_keys, clusterAssment, 20, flags)
    # 运行分类算法
    print 'running xgboost ...'
    # tune_parameters('./data/social/data2.csv')
    run_xgboost('./data/social/data2.csv')
    
    print "end."
