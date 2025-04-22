# Spam-SMS-Detection
## Project objective:
The project identifies whether an SMS message is **ham (not spam)** or **spam** with the help of Natural Language Processing (NLP) and machine learning algorithms. It is trained with a labeled SMS message dataset and becomes able to categorize incoming messages depending on their content.
## Technologies used :
**Python Pandas** for data manipulation **Numpy** for numerical computation **Scikit-learn for** machine learning algorithms **Matplotlib/Seaborn** for data visualization.
## Code:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
## Output:
![Screenshot 2025-04-22 173451](https://github.com/user-attachments/assets/032449c1-4cb8-4288-a2fc-8c1ba064ccfe)
![Screenshot 2025-04-22 173507](https://github.com/user-attachments/assets/0d6db3ac-0dae-4a2a-9b97-0c21ec35ecb3)
## Colab Link:
```
https://colab.research.google.com/drive/1yaIHF_6UDb1AHtL46IUqY0bZs2QGJTyU?usp=sharing
```
