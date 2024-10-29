import joblib
import jieba

# 确保在predict.py中定义了与train.py中相同的chinese_tokenizer
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 载入之前保存的模型和Vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 确保jieba分词器加载完毕
jieba.initialize()

# 新文本
new_texts = ["对你无语", "你真棒", "这部电影太糟糕了"]

# 转换新文本为特征向量，注意使用transform而不是fit_transform
new_features = vectorizer.transform(new_texts)

# 使用模型进行情绪预测
new_predictions = model.predict(new_features)

# 类别映射
class_mapping = {0: "negative", 1: "neutral", 2: "positive"}

# 将预测结果转换回情绪标签
new_labels = [class_mapping[prediction] for prediction in new_predictions]

# 打印出新文本及其预测的情绪
for text, label in zip(new_texts, new_labels):
    print(f"Text: \"{text}\" is predicted as {label}")