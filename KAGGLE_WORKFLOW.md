# Kaggle クラウドワークフロー運用メモ

ローカルPCでのデータDL・学習禁止。全てクラウド（Kaggle / GitHub Actions）で実行。

## 基本フロー

```
ノートブック編集 → git push → gh workflow run → Kaggle上で自動実行 → 自動提出
```

1. `deep-past/deep-past-baseline.ipynb` を編集
2. `git add && git commit && git push`
3. `gh workflow run kaggle-push.yml -f notebook_dir=deep-past`
4. Kaggle上で自動実行 → 自動提出（Internet OFF + competition_sources）
5. Kaggle Submissionsタブでスコア確認

## ディレクトリ構成（コンペごと）

```
kaggle-competitions/
├── .github/workflows/
│   ├── kaggle-push.yml        # 汎用: 任意のnotebook_dirを指定してkaggle kernels push
│   └── signate-submit.yml     # 汎用: SIGNATE提出
├── scripts/
│   └── setup-credentials.sh   # 新デバイス用: Kaggle/SIGNATE認証セットアップ
├── deep-past/                 # コンペごとにディレクトリを切る
│   ├── kernel-metadata.json
│   └── deep-past-baseline.ipynb
└── KAGGLE_WORKFLOW.md         # このファイル
```

## kernel-metadata.json テンプレート

```json
{
  "id": "yasunorim/<kernel-slug>",
  "title": "<Title>",
  "code_file": "<notebook>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [],
  "competition_sources": ["<competition-slug>"],
  "kernel_sources": [],
  "model_sources": []
}
```

### 注意点
- `competition_sources` 指定 + `enable_internet: "false"` で**Notebook提出**（実行完了→自動スコアリング）。**Internet ONだと提出扱いにならない**
- `is_private: "true"` は提出用。メダル狙いの公開用は別Notebookを作って `"false"` にする
- **コンペルール同意が必要**: ブラウザで https://www.kaggle.com/competitions/<slug>/rules から Accept しないとデータがマウントされない

## データパスの罠

### competition_sources のマウント先
```
/kaggle/input/competitions/<competition-slug>/
```
**注意**: `/kaggle/input/<competition-slug>/` ではない！ `competitions/` サブディレクトリが入る。

### dataset_sources のマウント先
```
/kaggle/input/<dataset-slug>/
```
こちらは直下。

### デバッグ用: ディレクトリ確認コード
```python
from pathlib import Path
for item in sorted(Path('/kaggle/input').iterdir()):
    print(f'  {item.name}/')
    for sub in sorted(item.iterdir()):
        print(f'    {sub.name} ({sub.stat().st_size:,} bytes)')
```

## GitHub Secrets（kaggle-competitionsリポ）

| Secret | 内容 |
|---|---|
| `KAGGLE_USERNAME` | `yasunorim` |
| `KAGGLE_KEY` | Kaggle API key |
| `SIGNATE_TOKEN_B64` | `base64 < ~/.signate/signate.json` の出力 |

### 新デバイスでのセットアップ
```bash
# GitHub CLIでSecrets確認
gh secret list

# ローカル認証セットアップ（環境変数から）
export KAGGLE_USERNAME=yasunorim
export KAGGLE_KEY=<key>
export SIGNATE_TOKEN_B64=$(base64 -w0 < ~/.signate/signate.json)
bash scripts/setup-credentials.sh
```

## kaggle CLI パス（Windows）

```
C:\Users\fw_ya\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\kaggle.exe
```

## エラー調査

### kernels status 確認
```bash
kaggle kernels status yasunorim/<kernel-slug>
```
ステータス: `QUEUED` → `RUNNING` → `COMPLETE` or `ERROR`

### ログ取得（cp932エンコード問題の回避）

`kaggle kernels output` はWindowsでcp932エラーになる（アッカド語等の非ASCII文字）。
API直叩きで回避：

```python
import base64, json, urllib.request, pathlib

# APIキーは ~/.kaggle/kaggle.json から自動取得（聞かない）
creds = json.loads(pathlib.Path.home().joinpath('.kaggle/kaggle.json').read_text())
username = creds['username']
key = creds['key']
auth = base64.b64encode(f'{username}:{key}'.encode()).decode()

url = 'https://www.kaggle.com/api/v1/kernels/output?userName=yasunorim&kernelSlug=<kernel-slug>'
req = urllib.request.Request(url)
req.add_header('Authorization', f'Basic {auth}')

resp = urllib.request.urlopen(req)
outer = json.loads(resp.read().decode('utf-8'))
log = json.loads(outer['logNullable'])

for e in log:
    if e.get('stream_name') == 'stdout':
        print(e['data'], end='')
```

### レスポンス形式
- `kernels output` API は JSON（zipではない）
- 構造: `{"logNullable": "<escaped JSON array>", ...}`
- `logNullable` をさらに `json.loads` してストリーム配列を得る
- 各要素: `{"stream_name": "stdout"|"stderr", "time": float, "data": "..."}`

## コンペ固有メモ

### Deep Past Challenge（Akkadian → English）
- **slug**: `deep-past-initiative-machine-translation`
- **評価指標**: BLEU + chrF++（2指標の組み合わせ）
- **データ構造**:
  - train: `oare_id`, `transliteration`, `translation`（1561行）
  - test: `id`, `text_id`, `line_start`, `line_end`, `transliteration`（4行）
  - submission: `id`, `translation`
  - 補助データ: `OA_Lexicon_eBL.csv`, `eBL_Dictionary.csv`, `publications.csv`(580MB), `published_texts.csv`, etc.
- **ベースライン**: TF-IDF char n-gram nearest neighbor
- **kernel slug**: `yasunorim/deep-past-akkadian-baseline`
