import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score 


def train_and_evaluate_model():
    # 1. 讀取與準備資料
    df = pd.read_csv("traindata.csv")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['Event'], df['Emotion'], test_size=0.2, random_state=42, stratify=df['Emotion']
    )
    
    # 2. 特徵工程 (Vectorization)
    # 針對英文語境，以 "word" 為單位建立特徵向量
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 3. 模型訓練 (Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 4. 預測與評估
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # 計算 Top-1 與 Top-3(只要正確答案在前三名就算對) 準確率
    acc_top1 = accuracy_score(y_test, y_pred)
    acc_top3 = top_k_accuracy_score(y_test, y_prob, k=3, labels=model.classes_)
    
    print(f"Top-1 準確率: {acc_top1*100:.1f}%")
    print(f"Top-3 準確率: {acc_top3*100:.1f}%")
    
    return model, vectorizer

my_model, my_vectorizer = train_and_evaluate_model()

