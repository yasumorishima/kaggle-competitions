# Google Service Account セットアップ手順

GitHub ActionsからGoogle DriveのEXP実験結果にアクセスするためのセットアップ。
全て無料（課金有効化不要）。

## Step 1: Google Cloud プロジェクト作成

1. https://console.cloud.google.com/ を開く
2. 上部の「プロジェクトを選択」→「新しいプロジェクト」
3. プロジェクト名: `kaggle-exp-sync`（任意）
4. 「作成」をクリック

## Step 2: Drive API 有効化

1. 左メニュー「APIとサービス」→「ライブラリ」
2. 「Google Drive API」を検索
3. 「有効にする」をクリック

## Step 3: Service Account 作成

1. 左メニュー「IAMと管理」→「サービスアカウント」
2. 「サービスアカウントを作成」をクリック
3. 名前: `drive-sync`（任意）
4. 「作成して続行」→ ロールはスキップ → 「完了」

## Step 4: JSONキー取得

1. 作成したサービスアカウント（`drive-sync@kaggle-exp-sync.iam.gserviceaccount.com`）をクリック
2. 「キー」タブ → 「キーを追加」→「新しいキーを作成」
3. 「JSON」を選択 → 「作成」
4. JSONファイルがダウンロードされる（**このファイルの中身がGitHub Secretsに入る**）

## Step 5: Drive フォルダを Service Account に共有

1. Google Drive（https://drive.google.com）を開く
2. `マイドライブ/kaggle` フォルダを右クリック → 「共有」
3. Service Accountのメールアドレスを入力:
   `drive-sync@kaggle-exp-sync.iam.gserviceaccount.com`
   （Step 3で作成したアドレス。Google Cloud Consoleで確認可能）
4. 権限: 「閲覧者」で十分
5. 「送信」

## Step 6: フォルダIDを取得

1. Google Driveで `kaggle` フォルダを開く
2. ブラウザのURLを確認: `https://drive.google.com/drive/folders/XXXXXXXXX`
3. `XXXXXXXXX` の部分がフォルダID

## Step 7: GitHub Secrets に追加

以下の2つのSecretを追加（全リポジトリで使う）:

### kaggle-competitions リポジトリ
1. https://github.com/yasumorishima/kaggle-competitions/settings/secrets/actions
2. 「New repository secret」:
   - Name: `GOOGLE_SERVICE_ACCOUNT_KEY`
   - Value: Step 4でダウンロードしたJSONファイルの**中身をそのまま貼り付け**
3. もう1つ:
   - Name: `DRIVE_SHARED_FOLDER_ID`
   - Value: Step 6で取得したフォルダID

### signate-comp リポジトリ
同様に追加:
1. https://github.com/yasumorishima/signate-comp/settings/secrets/actions
2. `GOOGLE_SERVICE_ACCOUNT_KEY` と `DRIVE_SHARED_FOLDER_ID` を追加

### drivendata-comp リポジトリ
同様に追加:
1. https://github.com/yasumorishima/drivendata-comp/settings/secrets/actions
2. `GOOGLE_SERVICE_ACCOUNT_KEY` と `DRIVE_SHARED_FOLDER_ID` を追加

## 完了後の確認

```bash
# kaggle-competitions で動作確認
gh workflow run "EXP W&B Sync" \
  --repo yasumorishima/kaggle-competitions \
  -f comp=s6e3-churn \
  -f exp=EXP001 \
  -f memo="service account動作確認"
```
