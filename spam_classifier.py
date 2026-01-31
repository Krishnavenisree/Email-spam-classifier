import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
# The dataset is tab-separated with no header
data = pd.read_csv("spam.csv", encoding="latin-1")
# Keep only the useful columns and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to numbers: ham = 0, spam = 1
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})



# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], 
    data['label_num'], 
    test_size=0.2, 
    random_state=42
)

# 4. Convert text to numeric features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# 7. Test with your own input
while True:
    msg = input("\nEnter a message (or type 'exit'): ")
    if msg.lower() == 'exit':
        break
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    if prediction == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM (Not Spam)")
