from pathlib import Path

input_dir = Path('/kaggle/input')
for item in sorted(input_dir.iterdir()):
    print(f'{item.name}/')
    for sub in sorted(item.iterdir()):
        print(f'  {sub.name} ({sub.stat().st_size:,} bytes)')

import pandas as pd
import numpy as np

DATA_DIR = Path('/kaggle/input/competitions/deep-past-initiative-machine-translation')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print(f'Train: {train.shape}')
print(f'  Columns: {train.columns.tolist()}')
print(f'  Sample transliteration: {train["transliteration"].iloc[0][:80]}')
print(f'  Sample translation: {train["translation"].iloc[0][:80]}')
print()
print(f'Test: {test.shape}')
print(f'  Columns: {test.columns.tolist()}')
print()
print(f'Submission format: {sample_sub.shape}')
print(f'  Columns: {sample_sub.columns.tolist()}')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train['transliteration'] = train['transliteration'].fillna('')
train['translation'] = train['translation'].fillna('')
test['transliteration'] = test['transliteration'].fillna('')

vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=50000,
    sublinear_tf=True
)

train_tfidf = vectorizer.fit_transform(train['transliteration'])
test_tfidf = vectorizer.transform(test['transliteration'])

print(f'Train TF-IDF matrix: {train_tfidf.shape}')
print(f'Test TF-IDF matrix: {test_tfidf.shape}')

sims = cosine_similarity(test_tfidf, train_tfidf)
best_idx = sims.argmax(axis=1)
best_scores = sims.max(axis=1)

predictions = []
for i, (idx, score) in enumerate(zip(best_idx, best_scores)):
    pred = train['translation'].iloc[idx]
    predictions.append(pred)
    print(f'--- Test {i} (similarity: {score:.3f}) ---')
    print(f'  Test:  {test["transliteration"].iloc[i][:100]}')
    print(f'  Match: {train["transliteration"].iloc[idx][:100]}')
    print(f'  Pred:  {pred[:100]}')
    print()

submission = pd.DataFrame({
    'id': test['id'],
    'translation': predictions
})
submission['translation'] = submission['translation'].fillna('unknown')

print(submission)
print()

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f'Saved submission.csv ({submission.shape})')

