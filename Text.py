import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import lightgbm as lgb
from sklearn.metrics import f1_score

'''train_data = pd.read_csv("train_set.csv", sep='\t')
test_data = pd.read_csv("test_a.csv", sep='\t')'''
train_data=pd.read_csv(r'D:\BaiduNetdiskDownload\新闻文本分类\train_set.csv',sep='\t')
test_data=pd.read_csv(r'D:\BaiduNetdiskDownload\新闻文本分类\test_a.csv',sep='\t')

plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.xlabel("类别")
plt.ylabel("数量")
plt.hist(train_data['label'], bins=100)
plt.show()

print(train_data['text'].apply(lambda x: len(x)).describe())
plt.hist(train_data['text'].apply(lambda x: len(x)), bins=1000)
plt.xlim(0,10000)
plt.show()



all_lines = ' '.join(list(train_data['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)
print(word_count[0])
print(word_count[1])
print(word_count[2])


plt.hist(train_data['text'].apply(lambda x: len(re.split('648|900|3750', x))), bins=1000)
plt.xlim(0,650)
plt.show()



'''def char_max(x):
    all_lines = ' '.join(list(x['text']))
    word_count = Counter(all_lines.split(" "))
    word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)
    print(x['label'].iloc[0], word_count[0])
train_data.groupby("label").apply(char_max)'''


tfidf = TfidfVectorizer()
train = tfidf.fit_transform(train_data['text'])
test = tfidf.transform(test_data['text'])

gbm = lgb.LGBMClassifier()
gbm.fit(train, train_data['label'])
pre = gbm.predict(test)


#print(f1_score(train_data['label']),average='macro')


df=pd.DataFrame()
df['label']=gbm.predict(test)
df.to_csv('submit1.csv',index=None)


