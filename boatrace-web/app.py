from flask import Flask, render_template, request, jsonify
import itertools
import sqlite3
import json
import os
import re
import urllib.request
from datetime import datetime

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            race_no INTEGER NOT NULL,
            predictions TEXT NOT NULL,
            result_1st INTEGER,
            result_2nd INTEGER,
            result_3rd INTEGER,
            payout INTEGER,
            purchase INTEGER,
            is_hit INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            nige_rate REAL,
            wind TEXT
        )
    ''')
    conn.commit()
    # 既存DBへのマイグレーション
    try:
        conn.execute('ALTER TABLE records ADD COLUMN nige_rate REAL')
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE records ADD COLUMN wind TEXT')
        conn.commit()
    except Exception:
        pass
    conn.close()

init_db()

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

WIND_OPTIONS = [
    "向かい風1m", "向かい風2m", "向かい風3m", "向かい風4m",
    "向かい風5m", "向かい風6m以上",
    "追い風1m", "追い風2m", "追い風3m", "追い風4m",
    "追い風5m", "追い風6m以上",
    "横風（左）1m", "横風（左）2m", "横風（左）3m以上",
    "横風（右）1m", "横風（右）2m", "横風（右）3m以上",
    "無風", "その他"
]

# 会場プロファイル
# in_rate: 1コース逃げのしやすさ補正（高いほどイン有利）
# upset:   荒れやすさ補正（高いほど外枠を加点）
# wind:    風の影響度（高いほど風向きの影響が大きい）
VENUE_PROFILES = {
    "桐生":  {"in_rate": 0.5,  "upset": 0.3, "wind": 0.4},
    "戸田":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.3},
    "江戸川":{"in_rate": -0.5, "upset": 1.5, "wind": 1.5},  # 最も荒れやすく風影響大
    "平和島":{"in_rate": 0.0,  "upset": 0.8, "wind": 0.5},
    "多摩川":{"in_rate": 0.5,  "upset": 0.3, "wind": 0.2},
    "浜名湖":{"in_rate": 0.0,  "upset": 0.8, "wind": 1.2},  # 風影響大
    "蒲郡":  {"in_rate": 0.5,  "upset": 0.3, "wind": 0.3},
    "常滑":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.4},
    "津":    {"in_rate": 0.3,  "upset": 0.4, "wind": 0.5},
    "三国":  {"in_rate": -0.3, "upset": 1.0, "wind": 1.0},  # 荒れやすい・風影響
    "びわこ":{"in_rate": 0.0,  "upset": 0.6, "wind": 0.8},
    "住之江":{"in_rate": 1.2,  "upset": 0.1, "wind": 0.1},  # 最もイン有利・堅い
    "尼崎":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.2},
    "鳴門":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.8},
    "丸亀":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.7},
    "児島":  {"in_rate": 0.5,  "upset": 0.4, "wind": 0.5},
    "宮島":  {"in_rate": 0.3,  "upset": 0.6, "wind": 0.6},
    "徳山":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.3},
    "下関":  {"in_rate": 0.5,  "upset": 0.4, "wind": 0.5},
    "若松":  {"in_rate": 0.3,  "upset": 0.6, "wind": 0.7},
    "芦屋":  {"in_rate": 0.8,  "upset": 0.2, "wind": 0.3},
    "福岡":  {"in_rate": 0.5,  "upset": 0.4, "wind": 0.4},
    "唐津":  {"in_rate": 0.3,  "upset": 0.5, "wind": 0.6},
    "大村":  {"in_rate": 1.5,  "upset": 0.0, "wind": 0.1},  # 最もイン勝率高い
}

# 会場×コース別1着率（直近3ヶ月実績、単位：%）
VENUE_COURSE_RATES = {
    "桐生":  [49.9, 14.4, 11.5, 14.1,  9.0,  2.2],
    "戸田":  [39.6, 17.9, 19.4, 14.6,  7.2,  2.3],
    "江戸川":[48.8, 14.7, 16.6, 12.6,  5.8,  2.8],
    "平和島":[49.1, 15.0, 16.5, 11.0,  7.4,  2.3],
    "多摩川":[54.0, 14.9, 15.1,  9.4,  5.6,  1.6],
    "浜名湖":[55.2, 13.4, 14.2, 10.6,  5.5,  2.1],
    "蒲郡":  [56.4, 12.3, 10.4, 13.7,  5.9,  2.1],
    "常滑":  [55.7, 13.8, 10.7, 11.1,  6.6,  2.5],
    "津":    [58.1, 13.8, 12.4, 10.0,  5.7,  0.7],
    "三国":  [49.1, 17.9, 15.0,  9.9,  7.2,  2.3],
    "びわこ":[52.5, 13.9, 16.3,  9.4,  7.0,  1.6],
    "住之江":[61.8, 11.6, 11.0, 10.4,  4.5,  1.4],
    "尼崎":  [61.4, 13.5, 10.9,  8.5,  5.0,  1.4],
    "鳴門":  [47.2, 15.2, 17.0, 10.0,  8.2,  3.2],
    "丸亀":  [59.1, 13.5, 11.6,  8.0,  6.7,  2.0],
    "児島":  [57.8, 11.0, 13.2, 10.6,  5.9,  2.2],
    "宮島":  [58.1, 12.5, 12.6, 10.7,  5.6,  1.7],
    "徳山":  [65.0, 10.7, 11.1,  7.9,  4.5,  1.2],
    "下関":  [64.5, 10.0, 11.6,  7.9,  5.4,  1.3],
    "若松":  [59.3, 15.7, 11.3,  8.2,  4.1,  1.9],
    "芦屋":  [61.8,  9.9, 10.0, 10.1,  7.3,  2.3],
    "福岡":  [62.0, 14.0, 14.8,  6.4,  2.5,  1.0],
    "唐津":  [57.9, 15.1, 11.4,  9.3,  5.7,  1.6],
    "大村":  [59.0, 13.0, 10.7, 10.5,  6.7,  1.3],
}

# 全会場平均（会場未選択時のフォールバック用）
DEFAULT_COURSE_RATES = [54.6, 13.9, 13.8, 10.8,  6.2,  2.0]

# 風向きの向かい風強度（強いほど外枠有利）
WIND_UPSET_BONUS = {
    "向かい風1m": 0.0, "向かい風2m": 0.1, "向かい風3m": 0.3,
    "向かい風4m": 0.5, "向かい風5m": 0.8, "向かい風6m以上": 1.2,
    "追い風1m": 0.0,  "追い風2m": 0.0,  "追い風3m": 0.1,
    "追い風4m": 0.2,  "追い風5m": 0.3,  "追い風6m以上": 0.5,
    "横風（左）1m": 0.2, "横風（左）2m": 0.4, "横風（左）3m以上": 0.7,
    "横風（右）1m": 0.2, "横風（右）2m": 0.4, "横風（右）3m以上": 0.7,
    "無風": 0.0, "その他": 0.0,
}

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def predict(boats, kimari=None, venue=None, wind=None, nige_rate=None):
    # 会場プロファイル取得
    vp = VENUE_PROFILES.get(venue, {"in_rate": 0.0, "upset": 0.5, "wind": 0.5})
    wind_bonus = WIND_UPSET_BONUS.get(wind, 0.0)
    # 風影響度 = 会場の風感度 × 風の強さ
    wind_effect = vp["wind"] * wind_bonus

    scores = []
    for b in boats:
        course = int(b.get('course', b.get('boat_number', 1)))
        et = safe_float(b.get('exhibit_time', 0))
        avg_st = safe_float(b.get('avg_st', 0))
        tilt = safe_float(b.get('tilt', 0))
        is_f = b.get('is_f', False)
        win1 = safe_float(b.get('win1_rate', 0))
        win2 = safe_float(b.get('win2_rate', 0))
        motor_win1 = safe_float(b.get('motor_win1', 0))
        motor_contrib = safe_float(b.get('motor_contrib', 0))

        # Base score: lower exhibit time = better, lower avg ST = better
        base = 0.0
        if et > 0:
            base -= (et - 6.7) * 10
        if avg_st > 0:
            base -= (avg_st - 0.15) * 8
        # Tilt penalty
        base -= tilt * 2
        # F持ちペナルティ
        if is_f:
            base -= 1.5

        # Player bonus
        player_bonus = win1 * 0.3 + win2 * 0.1

        # Motor bonus
        motor_bonus = motor_win1 * 0.15 + motor_contrib * 0.05

        score = base + player_bonus + motor_bonus

        # 会場×コース別1着率をベーススコアに使用（%→スケール変換）
        course_rates = VENUE_COURSE_RATES.get(venue, DEFAULT_COURSE_RATES)
        # 1着率をそのままスコアに（50%=+5, 10%=+1 程度のスケール）
        score += course_rates[course - 1] * 0.1

        # 荒れやすい場・向かい風: 外枠（4〜6コース）を加点
        upset_effect = vp["upset"] + wind_effect
        if course == 4:
            score += upset_effect * 0.4
        elif course == 5:
            score += upset_effect * 0.5
        elif course == 6:
            score += upset_effect * 0.3

        scores.append({'boat': b, 'score': score, 'course': course})

    scores.sort(key=lambda x: x['score'], reverse=True)

    def get_boat_by_course(course_num):
        for s in scores:
            if s['course'] == course_num:
                return s
        return None

    # 決まり手データから2着重み付けを取得
    nige_2nd_rates = {}
    top_course = scores[0]['course']
    if kimari and 'sim' in kimari:
        sim = kimari['sim']
        if top_course in sim and 'nige_2nd_rates' in sim[top_course]:
            nige_2nd_rates = sim[top_course]['nige_2nd_rates']

    # スコア順に全6艇を並べた候補リストを生成（最大12候補）
    candidates = []
    seen = set()
    for s1 in scores[:4]:
        bn1 = s1['boat']['boat_number']
        # 2着候補: 決まり手データがあれば優先、なければスコア順
        if nige_2nd_rates and s1['course'] == top_course:
            sorted_2nd_courses = [int(k) for k, _ in sorted(nige_2nd_rates.items(), key=lambda x: x[1], reverse=True)]
            seconds = [get_boat_by_course(c) for c in sorted_2nd_courses if get_boat_by_course(c)]
            seconds += [s for s in scores if s['boat']['boat_number'] != bn1 and s not in seconds]
        else:
            seconds = [s for s in scores if s['boat']['boat_number'] != bn1]
        for s2 in seconds[:4]:
            bn2 = s2['boat']['boat_number']
            for s3 in scores:
                bn3 = s3['boat']['boat_number']
                if bn3 in (bn1, bn2):
                    continue
                combo = f"{bn1}-{bn2}-{bn3}"
                if combo not in seen:
                    seen.add(combo)
                    # 複合スコア = 1着スコア×3 + 2着スコア×2 + 3着スコア
                    combined = s1['score'] * 3 + s2['score'] * 2 + s3['score']
                    candidates.append({'combo': combo, 'combined': combined, 'type': None})
                if len(candidates) >= 60:
                    break
            if len(candidates) >= 60:
                break
        if len(candidates) >= 60:
            break

    candidates.sort(key=lambda x: x['combined'], reverse=True)

    # スコアベースで仮タイプ付与（オッズなし時のフォールバック）
    # 上位から 本命×2, 対抗×3, 中穴×1 の6点
    type_labels = ['本命', '本命', '対抗', '対抗', '対抗', '中穴']
    results = []
    for i, c in enumerate(candidates[:len(type_labels)]):
        results.append({'combo': c['combo'], 'type': type_labels[i], 'combined': round(c['combined'], 2)})

    return {
        'predictions': results,
        'candidates': [{'combo': c['combo'], 'combined': round(c['combined'], 2)} for c in candidates[:60]],
        'score_order': [{'course': s['course'], 'boat_number': s['boat']['boat_number'], 'score': round(s['score'], 2)} for s in scores],
        'chaos': calc_chaos(scores, boats, vp, wind_effect, nige_rate)
    }


def calc_chaos(scores, boats, vp, wind_effect, nige_rate=None):
    reasons = []
    chaos = 0.0

    # イン逃げ率
    if nige_rate is not None:
        if nige_rate < 40:
            chaos += 3.0
            reasons.append(f"イン逃げ率が極めて低い({nige_rate:.1f}%)")
        elif nige_rate < 50:
            chaos += 1.5
            reasons.append(f"イン逃げ率が低い({nige_rate:.1f}%)")

    # スコア差：1位と2位の差が小さいほど荒れやすい
    if len(scores) >= 2:
        score_gap = scores[0]['score'] - scores[1]['score']
        if score_gap < 1.0:
            chaos += 2.5
            reasons.append("上位艇のスコア差が僅差")
        elif score_gap < 2.0:
            chaos += 1.0

    # 4〜6コースが上位スコアにいる
    outer_top = [s for s in scores[:3] if s['course'] >= 4]
    if len(outer_top) >= 2:
        chaos += 2.0
        reasons.append("外枠艇が上位スコアに複数")
    elif len(outer_top) == 1:
        chaos += 1.0
        reasons.append(f"{outer_top[0]['course']}コースが上位スコア")

    # 1コースのF持ち
    for b in boats:
        course = int(b.get('course', b.get('boat_number', 1)))
        if course == 1 and b.get('is_f', False):
            chaos += 2.5
            reasons.append("1コースにF持ち艇")
            break

    # 会場の荒れやすさ
    upset = vp.get('upset', 0.5)
    if upset >= 1.2:
        chaos += 2.0
        reasons.append("荒れやすい会場")
    elif upset >= 0.8:
        chaos += 1.0

    # 風の影響
    if wind_effect >= 0.8:
        chaos += 1.5
        reasons.append("風の影響が強い")
    elif wind_effect >= 0.4:
        chaos += 0.5

    # 最大10点でクランプ
    chaos = min(chaos, 10.0)
    level = 'low'
    label = '低め'
    if chaos >= 7:
        level = 'high'; label = '高い⚠️'
    elif chaos >= 4:
        level = 'mid'; label = 'やや高め'

    return {
        'score': round(chaos, 1),
        'level': level,
        'label': label,
        'reasons': reasons
    }


@app.route('/')
def index():
    return render_template('index.html', venues=VENUES, wind_options=WIND_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    boats = data.get('boats', [])
    kimari = data.get('kimari', None)
    venue = data.get('venue', None)
    wind = data.get('wind', None)
    nige_rate = data.get('nige_rate', None)
    result = predict(boats, kimari, venue=venue, wind=wind, nige_rate=nige_rate)
    return jsonify(result)


@app.route('/save_record', methods=['POST'])
def save_record():
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        INSERT INTO records (race_date, venue, race_no, predictions, created_at, nige_rate, wind)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('race_date', ''),
        data.get('venue', ''),
        data.get('race_no', 0),
        json.dumps(data.get('predictions', []), ensure_ascii=False),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data.get('nige_rate'),
        data.get('wind')
    ))
    conn.commit()
    record_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.close()
    return jsonify({'success': True, 'id': record_id})


VENUE_CODES = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04",
    "多摩川": "05", "浜名湖": "06", "蒲郡": "07", "常滑": "08",
    "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16",
    "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24",
}

@app.route('/debug_html', methods=['POST'])
def debug_html():
    """boatrace.jp から生HTML を取得して重要部分だけ返すデバッグ用エンドポイント"""
    data = request.get_json()
    venue = data.get('venue', '')
    race_no = data.get('race_no', 1)
    race_date = data.get('race_date', '')
    jcd = VENUE_CODES.get(venue)
    if not jcd:
        return jsonify({'error': '会場コード不明'})
    hd = race_date.replace('-', '')
    url = 'https://www.boatrace.jp/owpc/pc/race/raceresult?rno=' + str(race_no) + '&jcd=' + jcd + '&hd=' + hd
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'ja,en;q=0.5',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return jsonify({'error': str(e), 'url': url})

    # boatColorが登場する前後50文字のスニペット
    snippets = []
    for m in re.finditer(r'boatColor', html):
        start = max(0, m.start() - 20)
        end = min(len(html), m.end() + 80)
        snippets.append(html[start:end].replace('\n', ' ').replace('\r', ''))
        if len(snippets) >= 10:
            break

    # 3連単セクション
    trio_idx = html.find('3連単')
    trio_snippet = html[trio_idx:trio_idx+400].replace('\n', ' ') if trio_idx >= 0 else '(3連単 not found)'

    return jsonify({
        'url': url,
        'html_length': len(html),
        'boatcolor_snippets': snippets,
        'trio_section': trio_snippet,
    })


@app.route('/fetch_result', methods=['POST'])
def fetch_result():
    data = request.get_json()
    venue = data.get('venue', '')
    race_no = data.get('race_no', 1)
    race_date = data.get('race_date', '')  # YYYY-MM-DD

    jcd = VENUE_CODES.get(venue)
    if not jcd:
        return jsonify({'success': False, 'error': '会場コードが見つかりません'})

    hd = race_date.replace('-', '')
    url = 'https://www.boatrace.jp/owpc/pc/race/raceresult?rno=' + str(race_no) + '&jcd=' + jcd + '&hd=' + hd

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en;q=0.5',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return jsonify({'success': False, 'error': '取得失敗: ' + str(e), 'url': url})

    boats_found = []
    payout = 0

    # --- 着順取得 ---
    # 実際のHTML: <td class="is-fs14 is-fBold is-boatColor1">1</td>
    # boatColor の数字（色番号）ではなく td 内テキストが艇番
    m_boats = re.findall(r'boatColor\d+">\s*([1-6])\s*</td>', html)
    if len(m_boats) >= 3:
        boats_found = [int(x) for x in m_boats[:3]]

    # フォールバック: クラス末尾の " が省略されているケース
    if len(boats_found) < 3:
        m_boats2 = re.findall(r'boatColor\d+[^>]*>\s*([1-6])\s*</td>', html)
        if len(m_boats2) >= 3:
            boats_found = [int(x) for x in m_boats2[:3]]

    if len(boats_found) < 3:
        return jsonify({
            'success': False,
            'error': '着順データが見つかりません。手動で入力してください。',
            'url': url,
        })

    r1, r2, r3 = boats_found[0], boats_found[1], boats_found[2]

    # --- 払戻金取得 ---
    # 実際のHTML構造: 3連単セクションにnumberSet1_number spanが並び、
    # その後の td に払戻金額が入っている
    # 例: <span class="numberSet1_number is-type1">1</span>...<span is-type6>6</span>...<span is-type2>2</span>
    # 金額は別の td に "2,340" 形式で入る

    # 3連単セクションを切り出す
    payout_debug = ''
    trio_idx = html.find('3連単')
    if trio_idx < 0:
        trio_idx = html.find('三連単')

    if trio_idx >= 0:
        # 3連単から次の券種（3連複/2連単/単勝など）までを抽出
        trio_end = len(html)
        for marker in ['3連複', '2連単', '2連複', '単勝', '複勝', '拡連複']:
            idx = html.find(marker, trio_idx + 5)
            if 0 < idx < trio_end:
                trio_end = idx
        trio_html = html[trio_idx:trio_end]
        payout_debug = trio_html[:800].replace('\n', '').replace('\r', '')

        # 3連単セクション内の3桁以上の数字を探す（艇番1-6は除外）
        amounts_in_trio = re.findall(r'>(\d[\d,]*)<', trio_html)
        for a in amounts_in_trio:
            try:
                val = int(a.replace(',', ''))
                if val >= 100:  # 払戻金は最低100円以上
                    payout = val
                    break
            except Exception:
                continue

    # フォールバック: is-payout / is-pay クラスから取得
    if payout == 0:
        for pat in [r'is-payout\d*[^>]*>([\d,]+)', r'is-pay[^>]*>[\s¥]*([\d,]+)']:
            pm = re.search(pat, html)
            if pm:
                try:
                    val = int(pm.group(1).replace(',', ''))
                    if val >= 100:
                        payout = val
                        break
                except Exception:
                    pass

    return jsonify({
        'success': True,
        'r1': r1, 'r2': r2, 'r3': r3,
        'payout': payout,
        'payout_debug': payout_debug if payout == 0 else '',
        'url': url
    })


@app.route('/update_result', methods=['POST'])
def update_result():
    data = request.get_json()
    record_id = data.get('id')
    result_1st = data.get('result_1st')
    result_2nd = data.get('result_2nd')
    result_3rd = data.get('result_3rd')
    payout = data.get('payout', 0)
    purchase = data.get('purchase', 100)

    conn = get_db()
    row = conn.execute('SELECT predictions FROM records WHERE id=?', (record_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'success': False, 'error': 'Not found'})

    predictions = json.loads(row['predictions'])
    result_combo = f"{result_1st}-{result_2nd}-{result_3rd}"
    is_hit = 1 if any(p['combo'] == result_combo for p in predictions) else 0

    conn.execute('''
        UPDATE records SET result_1st=?, result_2nd=?, result_3rd=?,
        payout=?, purchase=?, is_hit=? WHERE id=?
    ''', (result_1st, result_2nd, result_3rd, payout, purchase, is_hit, record_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'is_hit': is_hit})


@app.route('/get_records')
def get_records():
    period = request.args.get('period', 'all')
    conn = get_db()
    if period == 'week':
        rows = conn.execute("SELECT * FROM records WHERE race_date >= date('now', '-7 days') ORDER BY id DESC").fetchall()
    elif period == 'month':
        rows = conn.execute("SELECT * FROM records WHERE race_date >= date('now', '-30 days') ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    conn.close()

    records = []
    for r in rows:
        records.append({
            'id': r['id'],
            'race_date': r['race_date'],
            'venue': r['venue'],
            'race_no': r['race_no'],
            'predictions': json.loads(r['predictions']),
            'result_1st': r['result_1st'],
            'result_2nd': r['result_2nd'],
            'result_3rd': r['result_3rd'],
            'payout': r['payout'],
            'purchase': r['purchase'],
            'is_hit': r['is_hit'],
            'created_at': r['created_at'],
            'nige_rate': r['nige_rate'],
            'wind': r['wind'],
        })
    return jsonify(records)


@app.route('/delete_record', methods=['POST'])
def delete_record():
    data = request.get_json()
    conn = get_db()
    conn.execute('DELETE FROM records WHERE id=?', (data.get('id'),))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/history')
def history():
    return render_template('history.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
