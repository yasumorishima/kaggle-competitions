キーワードやクリエイターで検索


メニュー
 投稿

見出し画像
【機械学習 第6回】LLM Classification で勉強

shogaku
shogaku
2025年11月26日 19:12

Kaggle LLM Classification Finetuning に挑戦：初心者の学習記録

Notebook

Japanese : LLM Classification
Explore and run machine learning code with Kaggle Notebooks |
www.kaggle.com



目次
Kaggle LLM Classification Finetuning に挑戦：初心者の学習記録
はじめに
コンペティション概要
タスク
データセット
評価指標
📦 1. ライブラリのインポート
📂 2. データの読み込み
データ構造
🔍 3. 探索的データ分析（EDA）

すべて表示


はじめに
Kaggle初心者として、「LLM Classification Finetuning」コンペティションに挑戦しました。

このコンペティションでは、Chatbot Arenaのデータを使用して、ユーザーがどのLLMの応答を好むかを予測します。この記事では、Notebookの実装の流れに沿って、学んだことや工夫した点を初心者目線でまとめています。

コンペティション概要
タスク
2つのLLMの応答を比較し、ユーザーがどちらを好むかを予測する3クラス分類：

winner_model_a: モデルAが好まれた

winner_model_b: モデルBが好まれた

winner_tie: 引き分け

データセット
トレーニングデータ: 57,477件

テストデータ: 予測対象

LLM: GPT-4, Claude, Llama 2, Gemini, Mistral など70以上

評価指標
Log Loss（対数損失）- 値が低いほど良い

📦 1. ライブラリのインポート
import numpy as np
import pandas as pd

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# 機械学習  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix

copy
学んだこと:

Plotlyでインタラクティブなグラフが作れる

XGBoost/LightGBMは勾配ブースティングの強力なライブラリ

MultiOutputClassifierで複数ターゲットを同時予測

📂 2. データの読み込み
train_df = pd.read_csv('/kaggle/input/llm-classification-finetuning/train.csv')
test_df = pd.read_csv('/kaggle/input/llm-classification-finetuning/test.csv')

copy
データ構造
id: 行ID

model_a, model_b: モデル名（⚠️テストデータにはない）

prompt: ユーザーの質問

response_a, response_b: 各モデルの応答

winner_model_a, winner_model_b, winner_tie: ターゲット

🔍 3. 探索的データ分析（EDA）
3.1 ターゲット変数の分布
target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
train_df['winner'] = train_df[target_cols].idxmax(axis=1)

copy
結果:

Model A 勝利: 20,064件（34.91%）

Model B 勝利: 19,652件（34.19%）

引き分け: 17,761件（30.90%）




学び: クラスがバランスしているため、特別なサンプリング不要

3.2 テキスト長の分析
train_df['prompt_length'] = train_df['prompt'].str.len()
train_df['response_a_length'] = train_df['response_a'].str.len()
train_df['response_b_length'] = train_df['response_b'].str.len()

copy
統計:

プロンプト平均: 369文字

レスポンスA平均: 1,378文字

レスポンスB平均: 1,386文字

📊 グラフ2: テキスト長の分布（4つのヒストグラム）

画像

学び: レスポンスの長さは重要な特徴量になる

3.3 モデルの使用頻度
Top 3:

gpt-4-1106-preview: 3,700回

gpt-3.5-turbo-0613: 3,500回

gpt-4-0613: 3,100回

📊 グラフ3: モデル使用頻度（横棒グラフ）

画像

学び: GPT-4やClaude系が頻繁に使用されている

3.4 モデル別勝率分析 ⭐
これが最重要な分析です！ 各モデルの強さを把握し、後で特徴量として活用します。

# モデル別勝率の計算
all_models_a = train_df.groupby('model_a')['winner_model_a'].agg(['sum', 'count'])
all_models_b = train_df.groupby('model_b')['winner_model_b'].agg(['sum', 'count'])

combined = pd.DataFrame({
    'wins': all_models_a['sum'].add(all_models_b['sum'], fill_value=0),
    'total': all_models_a['count'].add(all_models_b['count'], fill_value=0)
})
combined['win_rate'] = (combined['wins'] / combined['total'] * 100)

copy
Top 5（50回以上出現）:

gpt-4-1106-preview: 55.1%

gpt-3.5-turbo-0314: 54.6%

gpt-4-0125-preview: 51.4%

gpt-4-0314: 48.4%

claude-1: 43.9%

📊 グラフ4: モデル別勝率（棒グラフ）

画像

学び: GPT-4系が圧倒的に強い！この情報を特徴量に使う

3.5 WordCloud分析
all_prompts = ' '.join(train_df['prompt'].astype(str).values)
wordcloud = WordCloud(
    width=1200, height=600,
    background_color='white',
    stopwords=STOPWORDS,
    max_words=100
).generate(all_prompts)

copy
📊 グラフ5: WordCloud

画像


頻出単語: "write", "explain", "how", "create"などのタスク系単語

🔧 4. 特徴量エンジニアリング
合計330個の特徴量を作成します。

4.1 基本的な特徴量（26個）
def create_features(df):
    df = df.copy()
    
    # テキスト長
    df['prompt_length'] = df['prompt'].str.len()
    df['response_a_length'] = df['response_a'].str.len()
    df['response_b_length'] = df['response_b'].str.len()
    
    # 単語数
    df['prompt_words'] = df['prompt'].str.split().str.len()
    df['response_a_words'] = df['response_a'].str.split().str.len()
    df['response_b_words'] = df['response_b'].str.split().str.len()
    
    # 差と比率（ゼロ除算回避のため+1）
    df['length_diff'] = df['response_a_length'] - df['response_b_length']
    df['length_ratio'] = df['response_a_length'] / (df['response_b_length'] + 1)
    df['words_diff'] = df['response_a_words'] - df['response_b_words']
    df['words_ratio'] = df['response_a_words'] / (df['response_b_words'] + 1)
    
    # 平均単語長
    df['avg_word_length_a'] = df['response_a_length'] / (df['response_a_words'] + 1)
    df['avg_word_length_b'] = df['response_b_length'] / (df['response_b_words'] + 1)
    
    # テキスト特性
    df['punctuation_a'] = df['response_a'].str.count(r'[.,!?;:]')
    df['punctuation_b'] = df['response_b'].str.count(r'[.,!?;:]')
    df['uppercase_a'] = df['response_a'].str.count(r'[A-Z]')
    df['uppercase_b'] = df['response_b'].str.count(r'[A-Z]')
    df['digits_a'] = df['response_a'].str.count(r'\d')
    df['digits_b'] = df['response_b'].str.count(r'\d')
    df['newlines_a'] = df['response_a'].str.count(r'\n')
    df['newlines_b'] = df['response_b'].str.count(r'\n')
    
    # 構造的特徴（コード、リストの検出）
    df['has_code_a'] = df['response_a'].str.contains(r'```', regex=True).astype(int)
    df['has_code_b'] = df['response_b'].str.contains(r'```', regex=True).astype(int)
    df['has_list_a'] = df['response_a'].str.contains(r'\n\s*[•\-\*\d+\.]', regex=True).astype(int)
    df['has_list_b'] = df['response_b'].str.contains(r'\n\s*[•\-\*\d+\.]', regex=True).astype(int)
    
    return df

copy
重要テクニック:

+ 1でゼロ除算を回避

正規表現でコードブロックやリストを検出

構造化された応答（コード、リスト）は好まれる傾向

4.2 モデル情報の特徴量化（6個） ⭐
最も効果的な特徴量です！

model_win_rates = combined['win_rate'].to_dict()

def add_model_features(df):
    df = df.copy()
    
    if 'model_a' in df.columns and 'model_b' in df.columns:
        # トレーニングデータ: 実際のモデル情報を使用
        df['model_a_win_rate'] = df['model_a'].map(model_win_rates).fillna(35.0)
        df['model_b_win_rate'] = df['model_b'].map(model_win_rates).fillna(35.0)
        df['win_rate_diff'] = df['model_a_win_rate'] - df['model_b_win_rate']
        
        # モデルファミリーの分類
        def extract_family(m):
            if 'gpt-4' in m: return 'gpt4'
            elif 'gpt-3.5' in m: return 'gpt35'
            elif 'claude' in m: return 'claude'
            elif 'llama' in m: return 'llama'
            elif 'mistral' in m or 'mixtral' in m: return 'mistral'
            else: return 'other'
        
        df['model_a_family'] = df['model_a'].apply(extract_family)
        df['model_b_family'] = df['model_b'].apply(extract_family)
        df['same_family'] = (df['model_a_family'] == df['model_b_family']).astype(int)
    else:
        # テストデータ: デフォルト値
        df['model_a_win_rate'] = 35.0
        df['model_b_win_rate'] = 35.0
        df['win_rate_diff'] = 0.0
        df['model_a_family'] = 'other'
        df['model_b_family'] = 'other'
        df['same_family'] = 0
    
    return df

copy
⚠️ 重要な注意点: テストデータにはmodel_a/model_b情報がないため、条件分岐が必要です。

学び: ドメイン知識（どのモデルが強いか）を特徴量に組み込む威力

4.3 TF-IDF特徴量（300個）
テキストの内容を数値化します。

tfidf_prompt = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2),  # ユニグラムとバイグラム
    min_df=5             # 最低5回出現
)

prompt_tfidf_train = tfidf_prompt.fit_transform(train_df['prompt'])
prompt_tfidf_test = tfidf_prompt.transform(test_df['prompt'])

# response_a と response_b も同様に処理

copy
パラメータ解説:

max_features=100: 上位100単語

ngram_range=(1, 2): 単語とフレーズ（2単語）の両方

min_df=5: 稀な単語を除外

学び: TF-IDFはテキストを数値ベクトルに変換する標準手法

4.4 スパース行列の結合
メモリ効率のため、スパース行列として結合：

X_train = hstack([
    csr_matrix(X_train_num.values),  # 数値特徴量
    prompt_tfidf_train,               # スパース
    response_a_tfidf_train,           # スパース
    response_b_tfidf_train            # スパース
])

copy
最終特徴量: (57,477, 330)

学び: スパース行列で30万次元でもメモリ効率よく処理

🎯 5. モデル構築
4つのモデルを比較します。

5.1 データ分割
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, 
    stratify=train_df['winner']  # 層化抽出
)

copy
学び: stratifyでクラス比率を保ったまま分割

5.2 モデル1: ロジスティック回帰
lr_model = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, random_state=42, solver='saga', C=1.0)
)
lr_model.fit(X_train_split, y_train_split)

# 予測
y_val_pred_lr = lr_model.predict_proba(X_val_split)
y_val_pred_lr_array = np.column_stack([pred[:, 1] for pred in y_val_pred_lr])

# 評価
val_logloss_lr = log_loss(y_val_split, y_val_pred_lr_array)

copy
結果: Log Loss 1.0457

学び: MultiOutputClassifierで3つのターゲットを同時予測

5.3 モデル2: XGBoost
xgb_model = MultiOutputClassifier(
    xgb.XGBClassifier(
        n_estimators=200,    # ツリーの数
        max_depth=6,         # ツリーの深さ
        learning_rate=0.1,   # 学習率
        random_state=42,
        tree_method='hist'   # 高速化
    )
)
xgb_model.fit(X_train_split, y_train_split)

copy
結果: Log Loss 1.0003 ⭐

学び: XGBoostは表形式データに非常に強力

5.4 モデル3: LightGBM
lgb_model = MultiOutputClassifier(
    lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
)
lgb_model.fit(X_train_split, y_train_split)

copy
結果: Log Loss 1.0013

学び: LightGBMもXGBoost並みの性能

5.5 モデル4: ランダムフォレスト
rf_model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # 全CPUコアを使用
    )
)
rf_model.fit(X_train_split, y_train_split)

copy
結果: Log Loss 1.0180

5.6 モデル性能の比較
画像
📊 グラフ6: モデル性能比較（棒グラフ）

![モデル性能比較](./images/model_comparison.png)

copy
学び: 勾配ブースティング（XGBoost/LightGBM）が圧倒的

5.7 アンサンブル
# 重み付き平均
y_val_pred_ensemble = (
    y_val_pred_lr_array * 0.2 + 
    y_val_pred_xgb_array * 0.3 + 
    y_val_pred_lgb_array * 0.3 + 
    y_val_pred_rf_array * 0.2
)

val_logloss_ensemble = log_loss(y_val_split, y_val_pred_ensemble)

copy
結果: Log Loss 1.0028

学び: アンサンブルは安定するが、今回はXGBoost単体が最良

5.8 全データで再トレーニング
最終提出用に、全トレーニングデータでモデルを再トレーニングします。

lr_final = MultiOutputClassifier(LogisticRegression(...))
lr_final.fit(X_train, y_train)

xgb_final = MultiOutputClassifier(xgb.XGBClassifier(...))
xgb_final.fit(X_train, y_train)

lgb_final = MultiOutputClassifier(lgb.LGBMClassifier(...))
lgb_final.fit(X_train, y_train)

rf_final = MultiOutputClassifier(RandomForestClassifier(...))
rf_final.fit(X_train, y_train)

copy
学び: バリデーションは評価用、最終モデルは全データで学習

📤 6. 予測と提出
# 各モデルで予測
y_test_pred_lr = lr_final.predict_proba(X_test)
y_test_pred_xgb = xgb_final.predict_proba(X_test)
y_test_pred_lgb = lgb_final.predict_proba(X_test)
y_test_pred_rf = rf_final.predict_proba(X_test)

# アンサンブル
y_test_pred_array = (
    y_test_pred_lr_array * 0.2 + 
    y_test_pred_xgb_array * 0.3 + 
    y_test_pred_lgb_array * 0.3 + 
    y_test_pred_rf_array * 0.2
)

# 提出ファイル作成
submission = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': y_test_pred_array[:, 0],
    'winner_model_b': y_test_pred_array[:, 1],
    'winner_tie': y_test_pred_array[:, 2]
})

submission.to_csv('submission.csv', index=False)

copy
最終結果:

検証 Log Loss: 1.0003（XGBoost）

実際の提出スコア: 1.05812

学んだこと・気づき
✅ うまくいったこと
モデル情報の活用

各モデルの勝率を特徴量に組み込むことで大幅に改善

ドメイン知識の重要性を実感

勾配ブースティングの威力

XGBoost/LightGBMが他のモデルより圧倒的に優秀

デフォルトパラメータでも良好な結果

多様な特徴量

基本統計（26）+ モデル情報（6）+ TF-IDF（300）

構造的特徴（コード、リスト）の検出も有効

可視化の重要性

EDAで仮説を立て、それを特徴量に反映

Plotlyでインタラクティブなグラフが簡単に作れる

❌ 難しかったこと
テストデータの扱い

model_a/model_b情報がないことの対応

条件分岐でデフォルト値を設定

過学習の兆候

検証: 1.0003

実際: 1.05812

約5%の劣化が見られる

スパース行列の扱い

hstackとcsr_matrixの使い方に慣れが必要

メモリ効率は良いが、デバッグが難しい

今後の改善案
1. Cross-Validation
現在は単純なtrain/valid分割。より安定した評価のため5-fold CVを導入：

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # 各foldでモデルをトレーニング
    # 平均スコアで評価

copy
2. ハイパーパラメータチューニング
Optunaで自動最適化：

import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    model = xgb.XGBClassifier(**params)
    # 学習と評価
    return log_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

copy
3. より高度な特徴量
センチメント分析: TextBlobやVADERで感情スコア

可読性指標: Flesch Reading Ease Score

テキスト類似度: response_aとresponse_bのコサイン類似度

エンティティ認識: 人名、地名などの固有表現

4. 深層学習アプローチ（要GPU）
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = AutoModel.from_pretrained('microsoft/deberta-v3-base')

# テキストをembeddingに変換
embeddings = model(**tokenizer(text, return_tensors='pt'))

copy
まとめ
初心者として初めて取り組んだテキスト分類コンペティションでしたが、多くのことを学べました：

主な学び
特徴量エンジニアリングの重要性

単純な統計量からドメイン知識まで、様々な角度から特徴量を作成

モデル情報の活用が最も効果的だった

適切なモデル選択

XGBoost/LightGBMは表形式データに非常に強力

アンサンブルは必ずしも最良とは限らない

可視化による理解

EDAで仮説を立て、それを特徴量に反映

Plotlyで美しいグラフが簡単に作れる

実装スキル

スパース行列の扱い

MultiOutputClassifierの使用

効率的なコード構造化

技術スタック
データ処理: pandas, numpy

可視化: matplotlib, seaborn, plotly, wordcloud

機械学習: scikit-learn, xgboost, lightgbm

環境: Kaggle Notebook（CPU、Internet OFF）

次回への課題
この経験を活かして、次のコンペティションでは：

Cross-Validationの実装

ハイパーパラメータチューニング

深層学習への挑戦

より洗練された特徴量エンジニアリング

に挑戦していきたいと思います！

参考リソース
Chatbot Arena Paper

LMSYS Org

Kaggle Competition

scikit-learn Documentation

XGBoost Documentation

LightGBM Documentation

最後に: このコンペティションを通じて、機械学習の実践的なスキルを多く学ぶことができました。初心者の方にとって、この記事が参考になれば幸いです！

#kaggle
#特徴量
#XGBoost





shogaku
shogaku
GAS、VBA、Pythonを使って業務改善とかしています。備忘録や勉強メモを記載しています。



noteプレミアム
note pro
よくある質問・noteの使い方
プライバシー
クリエイターへのお問い合わせ
フィードバック
ご利用規約
通常ポイント利用特約
加盟店規約
資⾦決済法に基づく表⽰
特商法表記
投資情報の免責事項
【機械学習 第6回】LLM Classification で勉強｜shogaku