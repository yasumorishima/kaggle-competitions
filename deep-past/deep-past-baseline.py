# Auto-extracted from deep-past-baseline.ipynb

# %% Cell 1
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path('/kaggle/input/competitions/deep-past-initiative-machine-translation')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print(f'train: {train.shape}, columns: {train.columns.tolist()}')
print(f'test: {test.shape}, columns: {test.columns.tolist()}')
print(f'submission: {sample_sub.shape}, columns: {sample_sub.columns.tolist()}')

# %% Cell 2
# train: oare_id, transliteration, translation
# test: id, text_id, line_start, line_end, transliteration
# submission: id, translation

train['transliteration'] = train['transliteration'].fillna('')
train['translation'] = train['translation'].fillna('')
test['transliteration'] = test['transliteration'].fillna('')

# TF-IDF on transliteration (char n-grams for cuneiform transliteration)
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=50000,
    sublinear_tf=True
)

train_tfidf = vectorizer.fit_transform(train['transliteration'])
test_tfidf = vectorizer.transform(test['transliteration'])

print(f'Train TF-IDF: {train_tfidf.shape}')
print(f'Test TF-IDF: {test_tfidf.shape}')

# %% Cell 3
# Find nearest neighbor for each test row
sims = cosine_similarity(test_tfidf, train_tfidf)
best_idx = sims.argmax(axis=1)
best_scores = sims.max(axis=1)

predictions = []
for i, (idx, score) in enumerate(zip(best_idx, best_scores)):
    pred = train['translation'].iloc[idx]
    print(f'Test {i}: score={score:.3f}')
    print(f'  Test translit: {test["transliteration"].iloc[i][:80]}')
    print(f'  Match translit: {train["transliteration"].iloc[idx][:80]}')
    print(f'  Prediction: {pred[:80]}')
    print()
    predictions.append(pred)

# %% Cell 4
# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'translation': predictions
})
submission['translation'] = submission['translation'].fillna('unknown')

print(submission)
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f'\nSaved submission.csv ({submission.shape})')
