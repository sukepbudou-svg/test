# boatrace-predictor / PERRY AI プロジェクト概要

## 概要
競艇3連単予想ツール。LightGBMモデル＋複数エージェントで予想を生成し、
ローカルWeb（Flask + SQLite）に自動記録する。
ツール名: **PERRY AI**（黒船のペリーをコンセプトにしたAI予想キャラクター）

## 開発ブランチ
`claude/update-april-7-summary-ceGQV`

## 別PCや新セッションで再開するときの指示
```
boatrace-predictorプロジェクトの続きをお願いします。
リポジトリ: sukepbudou-svg/test
ブランチ: claude/update-april-7-summary-ceGQV
まずCLAUDE.mdを読んで現状を把握してから作業してください。
```

## Claudeへの運用ルール
- コード変更後に `git pull` を案内するときは、必ずその後に以下も伝えること：
  ```
  python main.py --mode auto
  ```
  を貼って予想を再スタートさせてください。
- ブラウザで `http://localhost:5001` を開くよう案内してください。

## 実行コマンド（Windows）
```
# プロジェクトフォルダへ移動
cd C:\Users\user\Desktop\test\boatrace-predictor

# 最新コードを取得
git pull

# 依存関係インストール（flask追加済み）
pip install -r requirements.txt

# 自動予想モード（Flask自動起動 → http://localhost:5001 で確認可）
python main.py --mode auto

# その他
python main.py --mode predict    # 一括予想（デバッグ用）
python main.py --mode backtest   # バックテスト
python main.py --mode train      # モデル再学習
```

### ワンクリック起動
デスクトップの `ボートレース予想起動.bat` をダブルクリック（git pull → auto自動実行）

---

## PERRY AI ローカルWeb（2026-08-15〜）

### アクセス
- `http://localhost:5001/` ― 本日の予想（60秒自動更新）
- `http://localhost:5001/stats` ― 統計ダッシュボード

### データベース
- `data/perry.db`（SQLite）
- スプレッドシートは不要（Google認証なしで動作）

### Web構成ファイル
| ファイル | 役割 |
|----------|------|
| `web/app.py` | Flaskアプリ本体・ルーティング |
| `web/database.py` | SQLite操作（保存・更新・集計） |
| `web/templates/index.html` | 本日の予想ページ |
| `web/templates/stats.html` | 統計ダッシュボード |

### API
| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/api/today` | GET | 本日の予想JSON |
| `/api/update_result` | POST | 結果更新（combination, payout） |
| `/api/pt_stats` | GET | PT帯別集計JSON |

---

## 現在の予想設定（2026-08-15 心機一転）

### ラベル（荒れPT基準）
| ラベル | 条件 | 参戦 | 色 |
|--------|------|------|----|
| **プチュン** | 荒れPT≥8 | ○ | 赤（#d93232）|
| **黒船熱** | 荒れPT=6〜7 | ○ | オレンジ（#d97c28）|
| **見送り** | 荒れPT≤5 | ✗ | グレー |

- 参戦ライン: **6P以上**
- 見送りでも予想を出力（データ収集のため）
- 出力は現在2点（主役艇固定 × MLモデル確率上位2点）← 将来3点以上に増やす可能性あり

### 買い目選出方式（hero1着固定 × 期待値上位2点 + 最低倍率フィルター） ← 点数変更時はここも更新すること
1. **主役艇（hero）選出**: 3〜6号艇から複合スコア最高の1艇
   - ST速さ (40%) + ET速さ (30%) + グレード (20%) + まくり実績 (10%)
   - 外枠ペナルティ: 4号艇-0.05 / 5号艇-0.10 / 6号艇-0.18
2. **2着・3着はMLモデル＋期待値で決定**:
   - 120通りの組み合わせ確率から「hero が1着」の組み合わせだけ抽出
   - ラベル別最低倍率フィルターを適用（コンセプト：荒れ狙い＝高倍率帯に絞る）
     - プチュン（荒れPT≥8）: **40倍以上**の組み合わせのみ
     - 黒船熱（荒れPT=6〜7）: **20倍以上**の組み合わせのみ
     - 見送り: フィルターなし（データ収集用）
   - フィルター通過後、**期待値（確率×倍率）上位2点**を買い目として採用
   - 該当組み合わせが0件の場合はそのレースをスキップ（予想なし）

> **重要**: 高倍率帯に絞ることで「荒れ狙いなのに本命寄りの買い目」という矛盾を解消。
> 倍率帯の結果を蓄積して、フィルター値の再調整を検討すること。

### 荒れPTスコアリング（src/model/predictor.py: _calc_arare_score）
3カテゴリ合計最大23点：

**① 1号艇の弱さ（最大9点）**
| 条件 | 点数 |
|------|------|
| 1号艇展示ST遅い（≥0.18） | +3 |
| 1号艇モーター2連率低い（<0.30） | +3 |
| 1号艇B級（grade≤2） | +2 |
| 1号艇展示タイム最遅 | +1 |

**② 外艇の脅威（最大8点）**
| 条件 | 点数 |
|------|------|
| 前付け（4〜6号が1〜2コース進入） | +3 |
| 外艇A1（1号がA1でない場合） | +2 |
| 外艇ST速い（≤0.12） | +2 |
| 外艇展示タイム最速（3〜6号） | +1 |

**③ 環境・条件（最大6点）**
| 条件 | 点数 |
|------|------|
| 風速≥7m | +2 |
| 波高≥15cm | +2 |
| 荒れ会場（江戸川+2、その他荒れ会場+1） | +1〜2 |
| 一般戦 | +1 |

---

## モデル構成
```
エージェント重み:
  ML（LightGBM）: 40%
  コース戦略:      25%
  選手成績:        20%
  モーター状態:    15%
```

## 主要ファイル
| ファイル | 役割 |
|----------|------|
| `src/model/predictor.py` | 予想エンジン・荒れ条件スコアリング・買い目選出 |
| `src/scheduler/auto_runner.py` | 発走時刻管理・自動ループ・DB書き込み |
| `web/app.py` | Flask Webアプリ |
| `web/database.py` | SQLite操作 |
| `src/agents/racer_performance.py` | 選手成績エージェント |
| `src/agents/course_strategy.py` | コース戦略エージェント |
| `src/agents/motor_form.py` | モーター状態エージェント |

## 注意事項
- Google Sheets APIは不要（sheets.pyはあるが `auto_runner.py` でオプション扱い）
- Sheetsを使う場合のみ `.env` に `SPREADSHEET_ID` と `GOOGLE_CREDENTIALS_PATH` が必要
- `--mode predict` は一括取得のため展示データが取れないことがある → 本番は `--mode auto` を使う
- Flask は `main.py --mode auto` 起動時に自動的にバックグラウンドスレッドで起動する
- データは `data/perry.db` に蓄積される
