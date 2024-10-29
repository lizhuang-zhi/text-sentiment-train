import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import jieba

# 从JSON文件加载数据
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

texts, labels = load_data('data.json')

# 使用jieba进行中文分词
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 文本预处理和特征提取
# 创建CountVectorizer，指定中文分词函数
vectorizer = CountVectorizer(tokenizer=chinese_tokenizer)
X = vectorizer.fit_transform(texts)

# 将标签转换成数值形式
class_mapping = {"negative": 0, "neutral": 1, "positive": 2}
y = [class_mapping[label] for label in labels]

# 分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 选择模型，这里使用朴素贝叶斯分类器
model = MultinomialNB()

# 训练模型
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# 保存模型和Vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")

