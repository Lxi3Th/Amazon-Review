import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


db = pd.read_csv('output.csv')

# For testing: verified_reviews
# For prediction: rating


ratingKey = {'1': 'Very-Bad', '2': 'Bad', '3': 'Good', '4': 'Very-Good', '5': 'Excellent'}

Xvar = db['verified_reviews']

Yvar = db['rating']

newdb = db.dropna(subset=['verified_reviews', 'rating'])

newX = newdb['verified_reviews']
newY = newdb['rating']



model = MultinomialNB()
wordToNumbers = CountVectorizer()

x_train, x_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3, random_state=42)

convert = wordToNumbers.fit_transform(x_train)

text = wordToNumbers.transform(x_test)

train = model.fit(convert, y_train)

predictConvert = model.predict(text)

score = accuracy_score(predictConvert, y_test)

print(f'Proof of accuracy: {score * 100:.2f}%')

def valueOfReview(msg, ratingKey):
    trainee = wordToNumbers.transform([msg])
    conv = model.predict(trainee)
    print(f'Your review: {msg}')
    print(f'Rating : {conv[0]} | Meaning: {ratingKey[conv[0]]}')


msg = 'I hate it worst product ever i dont suggest it'
valueOfReview(msg, ratingKey)

text = 'Super cool i recommend it so much 5 starts'
valueOfReview(text, ratingKey)