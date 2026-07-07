from flask import Flask, render_template, request, jsonify
import itertools
import sqlite3
import json
import os
import re
import urllib.request
import math
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
    try:
        conn.execute('ALTER TABLE records ADD COLUMN input_data TEXT')
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE records ADD COLUMN model_version TEXT')
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE records ADD COLUMN henkan TEXT')
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE records ADD COLUMN is_womens INTEGER DEFAULT 0')
        conn.commit()
    except Exception:
        pass
    conn.close()

init_db()

# モデルバージョン: 予想ロジックを変更したら必ず上げること
# v4: 2026-07-03 タイム符号バグ修正 + 対抗3シナリオ + 中穴廃止
# v5: 2026-07-04 選手個人の決まり手傾向による展開補正 → 効果なしと確定（採用見送り）
#     87レース検証: 本命-1.2pt/対抗±0。71レース再検証: v6と完全一致（0pt）。3回とも無効果
# v6: 2026-07-04 ハイブリッド対抗を採用。イン逃げ率65%以上は対抗3点を
#     1位艇1着固定（旧v3方式）、65%未満は3シナリオ維持。
#     バックテスト（得意会場×65%以上、18レース）: 対抗11.1%→27.8%で採用
# v7: 2026-07-05 本命2点目の3着を「逃がし3着率」最上位に差し替えて採用。
#     71レース検証: 本命+4.2pt/回収率+15.7pt（106%→121.7%）。2着バグ修正・v5は
#     3回の検証で一貫して無効果と判明したため不採用、3着差し替えのみ単体で採用
MODEL_VERSION = 'v7'
# v5ロジックのon/off: 3回の検証で無効果と確定。実戦・バックテストとも常時False
USE_V5_LIVE = False
# v6ハイブリッド対抗: 採用済み（2026-07-04）
USE_V6_LIVE = True
# v7: 本命2点目の3着差し替えのみ採用済み（2026-07-05）。2着バグ修正は無効果につき不採用
USE_V7_THIRD_LIVE = True

# 会場グループ（2026-07-04ユーザー設定。index.html/history.htmlの同名定数と同期すること）
TOKUI_VENUES = ['大村', '福岡', '桐生', '徳山']
KENSHO_VENUES = ['芦屋', '宮島', '常滑', '三国', '浜名湖', '尼崎', '若松', '蒲郡', '多摩川', '住之江', 'びわこ', '津', '丸亀', '児島']

TYPE_BY_INDEX_BASE = ['本命', '本命', '対抗', '対抗', '対抗', '万舟', '万舟']


def is_henkan_combo(combo, henkan_str):
    """返還艇を含むコンボかどうか判定（返還コンボは無効＝的中にも外れにも数えない）"""
    if not henkan_str:
        return False
    henkan_boats = {b.strip() for b in str(henkan_str).split(',') if b.strip()}
    return any(b in henkan_boats for b in combo.split('-'))


def _calc_recovery_base(with_result_rows):
    """本命+対抗5点ベースの回収率（万舟のまぐれ当たりを除外した実力値）。(recovery_base, race_count) を返す"""
    base_purchase = 0
    base_payout = 0
    base_races = 0
    for r in with_result_rows:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        henkan_str = r['henkan'] if 'henkan' in r.keys() else None
        base_combos = []
        for idx, p in enumerate(preds):
            t = p.get('type') or (TYPE_BY_INDEX_BASE[idx] if idx < len(TYPE_BY_INDEX_BASE) else '')
            if t in ('本命', '対抗') and p.get('combo') and not is_henkan_combo(p['combo'], henkan_str):
                base_combos.append(p['combo'])
        if not base_combos:
            continue
        base_races += 1
        base_purchase += 100 * len(base_combos)
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        if result_combo in base_combos and r['payout']:
            base_payout += r['payout']
    recovery_base = round(base_payout / base_purchase * 100, 1) if base_purchase > 0 else 0
    return recovery_base, base_races


def _judge_tier(venue, nige_rate, honmei_odds, venue_rows_by_venue):
    """予想画面のバナー判定（renderResult）と同じロジックをサーバー側で再現。
    成績記録のtier表示をバナーと一致させるための共通判定。"""
    if venue is None or nige_rate is None or honmei_odds is None:
        return None
    cond_ok = nige_rate >= 65 and honmei_odds >= 7
    if not cond_ok:
        return 'miokuri'
    if venue in TOKUI_VENUES:
        tier_base = 'shobu'
    elif venue in KENSHO_VENUES:
        tier_base = 'kensho'
    else:
        tier_base = 'other'

    rows = venue_rows_by_venue.get(venue, [])
    band_min = 65 if nige_rate >= 65 else math.floor(nige_rate / 5) * 5
    band_rows = [r for r in rows if r['nige_rate'] is not None and r['nige_rate'] >= band_min]
    rec_base, cnt = _calc_recovery_base(band_rows)
    if cnt < 10:
        rec_base, cnt = _calc_recovery_base(rows)

    if tier_base == 'shobu':
        if cnt >= 10 and rec_base < 80:
            return 'miokuri'
        return 'shobu'
    elif tier_base == 'kensho':
        if cnt >= 10 and rec_base >= 100:
            return 'shobu'
        return 'kensho'
    else:
        if cnt >= 20 and rec_base >= 120:
            return 'shobu'
        return 'miokuri'

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

def predict(boats, kimari=None, venue=None, wind=None, nige_rate=None, force_arekote=False, kimari_full=None, hybrid_taikou=False, v7_fix2nd=False, v7_third=False, extra_stats=None):
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
        lap = safe_float(b.get('lap', 0))
        avg_lap = safe_float(b.get('avg_lap', 0))
        straight = safe_float(b.get('straight', 0))
        avg_straight = safe_float(b.get('avg_straight', 0))
        mawariashi = safe_float(b.get('mawariashi', 0))
        avg_mawari = safe_float(b.get('avg_mawari', 0))
        player_class = b.get('player_class', '')

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

        # 級別ST修正能力補正（展示STが悪い時ほど級の差が出る）
        exhibit_st = safe_float(b.get('exhibit_st', 0))
        st_penalty = max(0, (exhibit_st - 0.15)) if exhibit_st > 0 else 0
        class_bonus = {'A1': 0.8, 'A2': 0.4, 'B1': 0.0, 'B2': -0.5}.get(player_class, 0.0)
        base += class_bonus * (1 + st_penalty * 10)

        # 周回タイム: 当日×0.6 + 平均×0.4 (小さいほど速い=良い)
        if lap > 0 and avg_lap > 0:
            combined_lap = lap * 0.6 + avg_lap * 0.4
            base += (36.0 - combined_lap) * 0.5
        elif lap > 0:
            base += (36.0 - lap) * 0.5

        # 直線タイム: 当日×0.6 + 平均×0.4 (小さいほど速い=良い)
        if straight > 0 and avg_straight > 0:
            combined_str = straight * 0.6 + avg_straight * 0.4
            base += (6.5 - combined_str) * 3.0
        elif straight > 0:
            base += (6.5 - straight) * 3.0

        # まわり足: 当日×0.6 + 平均×0.4 (小さいほど速い=良い)
        if mawariashi > 0 and avg_mawari > 0:
            combined_maw = mawariashi * 0.6 + avg_mawari * 0.4
            base += (5.0 - combined_maw) * 0.5
        elif mawariashi > 0:
            base += (5.0 - mawariashi) * 0.5

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

        scores.append({'boat': b, 'score': score, 'course': course, 'exhibit_st': exhibit_st})

    # ===== 展開補正: コース間のST比較で差し/まくりリスクを反映 =====
    # exhibit_stが入力されている艇だけ対象（0は未入力とみなす）
    st_by_course = {s['course']: s['exhibit_st'] for s in scores if s['exhibit_st'] > 0}

    if len(st_by_course) >= 2:
        def get_st(c): return st_by_course.get(c, 0.20)  # 未入力は遅め扱い

        st1 = get_st(1); st2 = get_st(2); st3 = get_st(3)
        st4 = get_st(4); st5 = get_st(5); st6 = get_st(6)

        for s in scores:
            c = s['course']
            adj = 0.0

            if c == 1:
                # 2コースとの差: 2が速いほどインの逃げリスク上昇
                sashi_threat = max(0, (st1 - st2) - 0.03)  # 0.03以上の差で差しリスク
                adj -= sashi_threat * 15
                # 3〜4コースがまくってきてもインは関係薄い
                # 2コースより内側の艇がいない場合は安全
                if st1 < st2 - 0.02:  # 1コースが明確に速い → 逃げ安全
                    adj += 0.5

            elif c == 2:
                # 1コースより速い → 差し期待値UP
                if st2 < st1 - 0.02:
                    adj += 0.6
                # 3コースが2より速い → 2コースは外から被される
                outer_threat = max(0, (st2 - st3) - 0.02)
                adj -= outer_threat * 10

            elif c == 3:
                # 2コース以内より速い → まくり期待値UP
                inner_avg = (st1 + st2) / 2
                if st3 < inner_avg - 0.03:
                    adj += 0.7
                # 4コースより遅い → 外から被される
                outer_threat = max(0, (st3 - st4) - 0.02)
                adj -= outer_threat * 8

            elif c == 4:
                # 3コース以内より速い → まくり期待値UP
                inner_avg = (st1 + st2 + st3) / 3
                if st4 < inner_avg - 0.04:
                    adj += 0.8
                # 5コースより遅い → 被される
                outer_threat = max(0, (st4 - st5) - 0.02)
                adj -= outer_threat * 6

            elif c == 5:
                inner_avg = (st2 + st3 + st4) / 3
                if st5 < inner_avg - 0.04:
                    adj += 0.6

            elif c == 6:
                inner_avg = (st3 + st4 + st5) / 3
                if st6 < inner_avg - 0.05:
                    adj += 0.5

            # 風の影響: 向かい風は外枠のまくりをさらに後押し
            if wind_effect > 0 and c >= 3:
                adj += wind_effect * 0.2

            s['score'] += adj

    # ===== v5 (2026-07-04): 選手個人の決まり手傾向による展開補正 =====
    # kimari_full: 各艇の差し率/捲り率/捲り差し率（直近6ヶ月優先、なければ直近1年）
    kdata = None
    if kimari_full and isinstance(kimari_full, dict):
        for period in ('直近6ヶ月', '直近1年'):
            if period in kimari_full and isinstance(kimari_full[period], dict):
                kdata = kimari_full[period]
                break
    if kdata:
        sashi_arr = kdata.get('sashi_active') or []
        maki_arr = kdata.get('maki_active') or []
        makis_arr = kdata.get('makis_active') or []

        def attack_rates(bn):
            """艇番2〜6の差し率/捲り率/捲り差し率（%）。配列indexは艇番-2"""
            i = bn - 2
            sa = safe_float(sashi_arr[i]) if 0 <= i < len(sashi_arr) else 0.0
            ma = safe_float(maki_arr[i]) if 0 <= i < len(maki_arr) else 0.0
            ms = safe_float(makis_arr[i]) if 0 <= i < len(makis_arr) else 0.0
            return sa, ma, ms

        for s in scores:
            bn = s['boat']['boat_number']
            if bn == 1:
                continue
            sa, ma, ms = attack_rates(bn)
            # 攻め力ボーナス: 決まり手を持つ艇は2着3着に絡みやすい（上限2.0）
            s['score'] += min(2.0, sa * 0.04 + ma * 0.05 + ms * 0.045)

        # 捲り屋がいる場合: その内側（1号艇以外）は沈みやすく、直外は連れて浮上
        for s in scores:
            bn = s['boat']['boat_number']
            if bn == 1:
                continue
            sa, ma, ms = attack_rates(bn)
            if ma + ms >= 15:  # 捲り系の合計が15%以上 → 捲り脅威とみなす
                for s2 in scores:
                    bn2 = s2['boat']['boat_number']
                    if bn2 == bn:
                        continue
                    if 1 < bn2 < bn:
                        s2['score'] -= 0.6  # 内側艇は展開で沈む
                    elif bn2 == bn + 1:
                        s2['score'] += 0.4  # 直外は展開が向く
            if sa >= 15:
                # 差し屋は伸び返しでイン残りを助ける（1号艇の2着残り）
                for s2 in scores:
                    if s2['boat']['boat_number'] == 1:
                        s2['score'] += 0.3

        # 1号艇の負けやすさ: 差され率＋捲られ率＋捲られ差され率が高いほど1着力を減点
        threat = (safe_float(kdata.get('sasar', 0))
                  + safe_float(kdata.get('makur_passive', 0))
                  + safe_float(kdata.get('makurS_passive', 0)))
        if threat > 0:
            for s in scores:
                if s['boat']['boat_number'] == 1:
                    s['score'] -= min(2.5, threat * 0.05)

    scores.sort(key=lambda x: x['score'], reverse=True)

    def get_boat_by_course(course_num):
        for s in scores:
            if s['course'] == course_num:
                return s
        return None

    # 決まり手データから2着重み付けを取得
    nige_2nd_rates = {}
    top_course = scores[0]['course']
    nige_3rd_rates = {}
    if kimari and 'sim' in kimari:
        sim = kimari['sim']
        if top_course in sim and 'nige_2nd_rates' in sim[top_course]:
            nige_2nd_rates = sim[top_course]['nige_2nd_rates']
        elif v7_fix2nd or v7_third:
            # v7: JSON経由だとsimのキーが文字列になり従来コードでは取得できていなかった
            entry = sim.get(str(top_course)) or {}
            nige_2nd_rates = entry.get('nige_2nd_rates') or {}
            nige_3rd_rates = entry.get('nige_3rd_rates') or {}

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

    # スコア→確率変換（softmax）: 期待値計算用の推定確率
    # 候補60点でほぼ全確率をカバーする前提。温度はスコアのばらつきで自動調整
    if candidates:
        import math as _math
        vals = [c['combined'] for c in candidates]
        mean_v = sum(vals) / len(vals)
        var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        temp = max(var_v ** 0.5, 1e-6)
        max_v = max(vals)
        exps = [_math.exp((v - max_v) / temp) for v in vals]
        total_exp = sum(exps)
        for c, e in zip(candidates, exps):
            c['prob'] = round(e / total_exp, 4)

    # 中穴予想（20〜99倍・独立モード）用のスコア補正: 超展開データの差し/まくり/まくり差し
    # 成功率（試行回数MIN_CHOTENKAI_TRIALS以上のみ信頼）で2・3着候補の「攻め力」を上乗せする。
    # combined/probはそのまま保持し、本命/対抗/裏熊の選定には一切影響させない
    ct_for_nakaana = (extra_stats or {}).get('cho_tenkai') if extra_stats else None
    nakaana_bonus = _nakaana_attack_bonus(ct_for_nakaana) if ct_for_nakaana else {}
    if nakaana_bonus:
        for c in candidates:
            parts = c['combo'].split('-')
            bn2, bn3 = int(parts[1]), int(parts[2])
            c['nakaana_score'] = round(c['combined'] + nakaana_bonus.get(bn2, 0.0) * 0.05 + nakaana_bonus.get(bn3, 0.0) * 0.03, 2)
    else:
        for c in candidates:
            c['nakaana_score'] = c['combined']

    # ===== 本命2点: スコア1位の艇が1着の上位2コンボ =====
    honmei = []
    top_bn = scores[0]['boat']['boat_number']
    for c in candidates:
        if c['combo'].startswith(f"{top_bn}-"):
            honmei.append(c)
        if len(honmei) >= 2:
            break

    # ===== v7 (2026-07-05検証用): 本命2点目の3着を「逃がし3着率」最上位に差し替え =====
    # 外れ方内訳で「3着だけズレ」が18.9%あった対策。1点目=スコア順の3着、
    # 2点目=統計上3着に来やすい艇、で3着の根拠を分散させる
    if v7_third and nige_3rd_rates and len(honmei) >= 2:
        h1_parts = honmei[0]['combo'].split('-')
        third_sorted = [int(k) for k, _ in sorted(nige_3rd_rates.items(), key=lambda x: x[1], reverse=True)]
        for c3 in third_sorted:
            s3 = get_boat_by_course(c3)
            if not s3:
                continue
            bn3 = str(s3['boat']['boat_number'])
            if bn3 in (h1_parts[0], h1_parts[1], h1_parts[2]):
                continue  # 1点目と同じ3着や1・2着と重複は不可
            new_combo = f"{h1_parts[0]}-{h1_parts[1]}-{bn3}"
            found = next((c for c in candidates if c['combo'] == new_combo), None)
            honmei[1] = found or {'combo': new_combo, 'combined': honmei[0]['combined'] - 0.01, 'prob': None}
            break

    # ===== 対抗3点 =====
    taikou = []
    used_combos = {c['combo'] for c in honmei}

    # --- v6ハイブリッド (2026-07-04): イン逃げ率65%以上は1位艇1着固定 ---
    # 勝負レース（堅い前提で選んだレース）では逆転シナリオを買わず、
    # 1位艇1着の2着3着バリエーションで固める（旧v3方式）
    if hybrid_taikou and nige_rate is not None and safe_float(nige_rate) >= 65:
        for c in candidates:
            if c['combo'].startswith(f"{top_bn}-") and c['combo'] not in used_combos:
                taikou.append(c)
                used_combos.add(c['combo'])
            if len(taikou) >= 3:
                break

    # --- 対抗1点目: スコア2位の艇が1着（展開逆転シナリオ）---
    if len(taikou) < 3 and len(scores) >= 2:
        second_bn = scores[1]['boat']['boat_number']
        for c in candidates:
            if c['combo'].startswith(f"{second_bn}-") and c['combo'] not in used_combos:
                taikou.append(c)
                used_combos.add(c['combo'])
                break

    # --- 対抗2点目: 1位1着 + STが最速の艇が2着（展示ST展開シナリオ）---
    if len(taikou) < 3:
        st_entries = [(s['course'], s['exhibit_st'], s['boat']['boat_number'])
                      for s in scores if s.get('exhibit_st', 0) > 0
                      and s['boat']['boat_number'] != top_bn]
        if st_entries:
            best_st_bn = min(st_entries, key=lambda x: x[1])[2]  # STが小さい=速い
            for c in candidates:
                parts = c['combo'].split('-')
                if parts[0] == str(top_bn) and parts[1] == str(best_st_bn) and c['combo'] not in used_combos:
                    taikou.append(c)
                    used_combos.add(c['combo'])
                    break

    # --- 対抗3点目: 残りスコア上位コンボから本命と被らないもの ---
    if len(taikou) < 3:
        for c in candidates:
            if c['combo'] not in used_combos:
                taikou.append(c)
                used_combos.add(c['combo'])
                break

    # 不足分はスコア順で補完
    for c in candidates:
        if len(honmei) >= 2 and len(taikou) >= 3:
            break
        if c['combo'] not in used_combos:
            if len(honmei) < 2:
                honmei.append(c)
            elif len(taikou) < 3:
                taikou.append(c)
            used_combos.add(c['combo'])

    results = []
    for c in honmei[:2]:
        results.append({'combo': c['combo'], 'type': '本命', 'combined': round(c['combined'], 2), 'prob': c.get('prob')})
    for c in taikou[:3]:
        results.append({'combo': c['combo'], 'type': '対抗', 'combined': round(c['combined'], 2), 'prob': c.get('prob')})

    chaos = calc_chaos(scores, boats, vp, wind_effect, nige_rate)

    # 裏熊モード: 強制フラグのときのみ（自動切り替えは廃止）
    if force_arekote:
        ura = predict_arekote_v3(scores, kimari_full=kimari_full, nige_rate=nige_rate,
                                  cho_tenkai=(extra_stats or {}).get('cho_tenkai') if extra_stats else None)
        return {
            'predictions': results,
            'candidates': [{'combo': c['combo'], 'combined': round(c['combined'], 2), 'prob': c.get('prob'), 'nakaana_score': c.get('nakaana_score')} for c in candidates[:60]],
            'score_order': [{'course': s['course'], 'boat_number': s['boat']['boat_number'], 'score': round(s['score'], 2)} for s in scores],
            'chaos': chaos,
            'arekote_mode': True,
            'arekote_predictions': ura['predictions'] if ura else [],
            'ura_judge': {k: v for k, v in ura.items() if k != 'predictions'} if ura else None,
        }

    return {
        'predictions': results,
        'candidates': [{'combo': c['combo'], 'combined': round(c['combined'], 2), 'prob': c.get('prob'), 'nakaana_score': c.get('nakaana_score')} for c in candidates[:60]],
        'score_order': [{'course': s['course'], 'boat_number': s['boat']['boat_number'], 'score': round(s['score'], 2)} for s in scores],
        'chaos': chaos,
    }


def _score_boat_group(b, venue, group):
    """単一要素モデル比較用（2026-07-07・検証段階）: 通常モデルの要素を
    グループA「当日の調子」（展示タイム・展示ST・チルト・周回/直線/まわり足）と
    グループB「地力・機材」（選手勝率・級別・モーター・会場×コース別1着率）に分け、
    片方のグループの要素だけでスコアを算出する。本命/対抗/裏熊/中穴のロジックには影響しない"""
    course = int(b.get('course', b.get('boat_number', 1)))
    score = 0.0
    if group == 'A':
        et = safe_float(b.get('exhibit_time', 0))
        avg_st = safe_float(b.get('avg_st', 0))
        exhibit_st = safe_float(b.get('exhibit_st', 0))
        tilt = safe_float(b.get('tilt', 0))
        is_f = b.get('is_f', False)
        lap = safe_float(b.get('lap', 0))
        avg_lap = safe_float(b.get('avg_lap', 0))
        straight = safe_float(b.get('straight', 0))
        avg_straight = safe_float(b.get('avg_straight', 0))
        mawariashi = safe_float(b.get('mawariashi', 0))
        avg_mawari = safe_float(b.get('avg_mawari', 0))
        if et > 0:
            score -= (et - 6.7) * 10
        # 複合モデルのベーススコアは「平均ST」を使っている（展示STではない）ため、
        # グループA単独でも両方のST情報を反映する（平均ST=通算の癖、展示ST=当日の実際の出足）
        if avg_st > 0:
            score -= (avg_st - 0.15) * 8
        if exhibit_st > 0:
            score -= (exhibit_st - 0.15) * 8
        score -= tilt * 2
        if is_f:
            score -= 1.5
        if lap > 0 and avg_lap > 0:
            score += (36.0 - (lap * 0.6 + avg_lap * 0.4)) * 0.5
        elif lap > 0:
            score += (36.0 - lap) * 0.5
        if straight > 0 and avg_straight > 0:
            score += (6.5 - (straight * 0.6 + avg_straight * 0.4)) * 3.0
        elif straight > 0:
            score += (6.5 - straight) * 3.0
        if mawariashi > 0 and avg_mawari > 0:
            score += (5.0 - (mawariashi * 0.6 + avg_mawari * 0.4)) * 0.5
        elif mawariashi > 0:
            score += (5.0 - mawariashi) * 0.5
    else:  # group == 'B'
        win1 = safe_float(b.get('win1_rate', 0))
        win2 = safe_float(b.get('win2_rate', 0))
        motor_win1 = safe_float(b.get('motor_win1', 0))
        motor_contrib = safe_float(b.get('motor_contrib', 0))
        player_class = b.get('player_class', '')
        class_bonus = {'A1': 0.8, 'A2': 0.4, 'B1': 0.0, 'B2': -0.5}.get(player_class, 0.0)
        score += class_bonus
        score += win1 * 0.3 + win2 * 0.1
        score += motor_win1 * 0.15 + motor_contrib * 0.05
        course_rates = VENUE_COURSE_RATES.get(venue, DEFAULT_COURSE_RATES)
        score += course_rates[course - 1] * 0.1
    return score


def predict_single_factor(boats, venue=None, group='A'):
    """単一要素モデル比較用の簡易予想（本命2点+対抗3点、複合スコアのみで選定。
    v6ハイブリッドやv7の3着差し替えなどは使わず、純粋にスコア順だけで選ぶ）"""
    scores = []
    for b in boats:
        course = int(b.get('course', b.get('boat_number', 1)))
        scores.append({'boat': b, 'score': _score_boat_group(b, venue, group), 'course': course})
    scores.sort(key=lambda x: x['score'], reverse=True)

    candidates = []
    seen = set()
    for s1 in scores[:4]:
        bn1 = s1['boat']['boat_number']
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
                    combined = s1['score'] * 3 + s2['score'] * 2 + s3['score']
                    candidates.append({'combo': combo, 'combined': combined})
                if len(candidates) >= 60:
                    break
            if len(candidates) >= 60:
                break
        if len(candidates) >= 60:
            break
    candidates.sort(key=lambda x: x['combined'], reverse=True)

    top_bn = scores[0]['boat']['boat_number']
    honmei = [c for c in candidates if c['combo'].startswith(f"{top_bn}-")][:2]
    used = {c['combo'] for c in honmei}
    taikou = []
    for c in candidates:
        if len(taikou) >= 3:
            break
        if c['combo'] not in used:
            taikou.append(c)
            used.add(c['combo'])

    results = []
    for c in honmei:
        results.append({'combo': c['combo'], 'type': '本命', 'combined': round(c['combined'], 2)})
    for c in taikou:
        results.append({'combo': c['combo'], 'type': '対抗', 'combined': round(c['combined'], 2)})
    return {'predictions': results}


MIN_CHOTENKAI_TRIALS = 3  # 超展開データの試行回数がこれ未満なら信頼度不足として使わない


def _wl_rate(pair):
    """[勝ち,試行] ペアから (成功率%, 試行数) を返す。データなし/試行0はNone"""
    if not pair or not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    w, t = pair
    if w is None or t is None or t <= 0:
        return None
    return (w / t * 100, t)


def _nakaana_attack_bonus(cho_tenkai):
    """超展開データ（差し/まくり/まくり差しの試行回数つき成功率）から、
    2〜6号艇それぞれの「攻め成功率」ボーナスを返す（中穴予想の2・3着候補の底上げに使用）。
    試行回数がMIN_CHOTENKAI_TRIALS未満の項目は信頼度不足として無視する。"""
    ct = cho_tenkai if isinstance(cho_tenkai, dict) else {}
    ct_maki = ct.get('makuri') or []
    ct_makis = ct.get('makurisashi') or []
    ct_sashi = ct.get('sashi') or []
    bonus = {}
    for bn in range(2, 7):
        i = bn - 2
        rates = []
        for arr in (ct_maki, ct_makis, ct_sashi):
            if 0 <= i < len(arr):
                wl = _wl_rate(arr[i])
                if wl and wl[1] >= MIN_CHOTENKAI_TRIALS:
                    rates.append(wl[0])
        if rates:
            bonus[bn] = max(rates)
    return bonus


def predict_arekote_v3(scores, kimari_full=None, nige_rate=None, cho_tenkai=None):
    """裏熊予想v3 (2026-07-05): 超展開データ（直接対戦の決まり手 勝ち/試行）を活用

    v2からの変更点:
      - 主役の攻撃力判定: cho_tenkaiの試行回数を伴う成功率を優先使用
        （試行回数MIN_CHOTENKAI_TRIALS未満は信頼度不足としてkimari_fullにフォールバック）
      - 1号艇の弱点条件に「抵抗」回数（攻められた際に粘れているか）を追加
      - 相手2艇の選定にも超展開データの差し成功率を反映

    出動条件（AND）:
      (1) イン逃げ率40%未満
      (2) 1号艇に弱点2つ以上（スコア3位以下/B級/展示ST5番手以下/差され捲られ率30%超/抵抗不足）
      (3) 仕留める主役がいる（3〜5コース・まくり系15%以上・STが1号艇より速い・スコア2位以内）
      (4) オッズ妙味（20倍未満は足切り）→ フロント側で判定
    買い目: 主役-相手2艇-全 のフォーメーション8点
    条件未達でも参考予想として買い目は返す（バナーで条件充足数を表示）
    """
    if not scores:
        return None
    boat1 = next((s for s in scores if s['course'] == 1), None)
    # scoresはスコア降順ソート済み
    rank_of = {s['boat']['boat_number']: i + 1 for i, s in enumerate(scores)}

    # 決まり手データ（直近6ヶ月優先）
    kdata = None
    if kimari_full and isinstance(kimari_full, dict):
        for period in ('直近6ヶ月', '直近1年'):
            if period in kimari_full and isinstance(kimari_full[period], dict):
                kdata = kimari_full[period]
                break
    maki_arr = (kdata or {}).get('maki_active') or []
    makis_arr = (kdata or {}).get('makis_active') or []
    ct = cho_tenkai if isinstance(cho_tenkai, dict) else {}
    ct_maki = ct.get('makuri') or []
    ct_makis = ct.get('makurisashi') or []
    ct_sashi = ct.get('sashi') or []

    def attack(bn):
        """超展開データがあれば試行回数つきの成功率を優先、無ければkimari_fullにフォールバック"""
        i = bn - 2
        maki_wl = _wl_rate(ct_maki[i]) if 0 <= i < len(ct_maki) else None
        makis_wl = _wl_rate(ct_makis[i]) if 0 <= i < len(ct_makis) else None
        has_reliable_ct = (
            (maki_wl and maki_wl[1] >= MIN_CHOTENKAI_TRIALS) or
            (makis_wl and makis_wl[1] >= MIN_CHOTENKAI_TRIALS)
        )
        if has_reliable_ct:
            ma = maki_wl[0] if maki_wl and maki_wl[1] >= MIN_CHOTENKAI_TRIALS else 0.0
            ms = makis_wl[0] if makis_wl and makis_wl[1] >= MIN_CHOTENKAI_TRIALS else 0.0
            return ma, ms
        ma = safe_float(maki_arr[i]) if 0 <= i < len(maki_arr) else 0.0
        ms = safe_float(makis_arr[i]) if 0 <= i < len(makis_arr) else 0.0
        return ma, ms

    # 条件1: イン逃げ率40%未満
    cond_nige = nige_rate is not None and safe_float(nige_rate) < 40

    # 条件2: 1号艇の弱点2つ以上
    weaknesses = []
    boat1_st = 0
    if boat1:
        bn1 = boat1['boat']['boat_number']
        boat1_st = boat1.get('exhibit_st', 0)
        if rank_of.get(bn1, 1) >= 3:
            weaknesses.append(f"スコア{rank_of[bn1]}位")
        pc = boat1['boat'].get('player_class', '')
        if pc in ('B1', 'B2'):
            weaknesses.append(f"{pc}級")
        st_sorted = sorted([s for s in scores if s.get('exhibit_st', 0) > 0], key=lambda x: x['exhibit_st'])
        if boat1_st > 0:
            st_rank1 = next((i + 1 for i, s in enumerate(st_sorted) if s['course'] == 1), None)
            if st_rank1 and st_rank1 >= 5:
                weaknesses.append(f"展示ST{st_rank1}番手")
        threat = (safe_float((kdata or {}).get('sasar', 0))
                  + safe_float((kdata or {}).get('makur_passive', 0))
                  + safe_float((kdata or {}).get('makurS_passive', 0)))
        if threat > 30:
            weaknesses.append(f"差され捲られ率{round(threat)}%")
        # 超展開データ: 「抵抗」回数が少ない = 攻められると粘れない
        teikou = ct.get('teikou')
        if teikou is not None and teikou <= 1:
            weaknesses.append(f"抵抗{teikou}回（粘れていない）")
    cond_boat1 = len(weaknesses) >= 2

    # 条件3: 仕留める主役（3〜5コース）
    shuyaku = None
    shuyaku_candidates = []
    for s in scores:
        if s['course'] not in (3, 4, 5):
            continue
        bn = s['boat']['boat_number']
        ma, ms = attack(bn)
        st_ok = s.get('exhibit_st', 0) > 0 and boat1_st > 0 and s['exhibit_st'] < boat1_st
        if (ma + ms) >= 15 and st_ok and rank_of.get(bn, 9) <= 2:
            shuyaku_candidates.append({'bn': bn, 'course': s['course'], 'maki': ma, 'makis': ms, 'score': s['score']})
    if shuyaku_candidates:
        shuyaku = max(shuyaku_candidates, key=lambda x: x['score'])
    cond_shuyaku = shuyaku is not None

    # 条件未達でも参考予想: 1号艇以外のスコア最上位を主役に
    if shuyaku is None:
        fallback = [s for s in scores if s['course'] != 1]
        if not fallback:
            return None
        fb = max(fallback, key=lambda x: x['score'])
        ma, ms = attack(fb['boat']['boat_number'])
        shuyaku = {'bn': fb['boat']['boat_number'], 'course': fb['course'], 'maki': ma, 'makis': ms, 'score': fb['score']}

    scenario = 'まくり' if shuyaku['maki'] >= shuyaku['makis'] else 'まくり差し'
    bn1 = boat1['boat']['boat_number'] if boat1 else 1

    # 相手2艇: シナリオ整合スコア + 超展開データの差し成功率で選ぶ
    def aite_score(s):
        sc = s['score']
        if s['course'] == shuyaku['course'] + 1:
            sc += 1.0  # 直外は展開が向く
        if scenario == 'まくり' and 1 < s['course'] < shuyaku['course']:
            sc -= 1.0  # 主役より内側は沈みやすい
        i = s['course'] - 2
        sashi_wl = _wl_rate(ct_sashi[i]) if 0 <= i < len(ct_sashi) else None
        if sashi_wl and sashi_wl[1] >= MIN_CHOTENKAI_TRIALS:
            sc += sashi_wl[0] * 0.05  # 差し成功率が高い艇を2着候補として底上げ
        return sc

    others = [s for s in scores if s['boat']['boat_number'] != shuyaku['bn']]
    if scenario == 'まくり':
        pool = [s for s in others if s['boat']['boat_number'] != bn1]  # まくり時は1号艇を2着に置かない
    else:
        pool = others
    pool_sorted = sorted(pool, key=aite_score, reverse=True)
    aite = [p['boat']['boat_number'] for p in pool_sorted[:2]]

    # フォーメーション: 主役-相手2艇-全（8点）
    combos = []
    for a in aite:
        for s in scores:
            b3 = s['boat']['boat_number']
            if b3 in (shuyaku['bn'], a):
                continue
            combos.append({'combo': f"{shuyaku['bn']}-{a}-{b3}", 'type': '裏熊'})

    return {
        'predictions': combos,
        'shuyaku': shuyaku['bn'],
        'scenario': scenario,
        'aite': aite,
        'formation': f"{shuyaku['bn']}-{aite[0]}{aite[1]}-全" if len(aite) >= 2 else '',
        'conds': {
            'nige': bool(cond_nige),
            'nige_val': safe_float(nige_rate) if nige_rate is not None else None,
            'boat1': bool(cond_boat1),
            'weaknesses': weaknesses,
            'shuyaku_ok': bool(cond_shuyaku),
            'clear': int(cond_nige) + int(cond_boat1) + int(cond_shuyaku),
            'total': 3,
        },
    }


def predict_arekote(scores, candidates, wind, wind_effect):
    """
    旧・荒れモード予想（v2に置き換え済み・未使用）
    万舟×2 + 裏熊×3（展開重視、1号艇除外）+ 裏熊×1（差し/まくり判断で1号艇2or3着）
    """
    # 万舟×2: 既存候補リストの最後尾（低スコア＝高倍率）上位2点
    # candidatesはスコア降順なので末尾から取る（ただし候補が少ない場合は後半から）
    manzoku = []
    seen_manzoku = set()
    for c in reversed(candidates):
        if c['combo'] not in seen_manzoku:
            manzoku.append({'combo': c['combo'], 'type': '万舟'})
            seen_manzoku.add(c['combo'])
        if len(manzoku) >= 2:
            break

    # 展開スコア算出（1号艇除外）
    # 追い風/横風: 3-4コース加点、向かい風: 2号艇加点
    wind_str = wind or ''
    is_mukai = '向かい風' in wind_str
    is_oi_or_yoko = any(w in wind_str for w in ['追い風', '横風'])

    outer_boats = [s for s in scores if s['course'] != 1]
    tenkai_scores = []
    for s in outer_boats:
        ts = s['score']
        c = s['course']
        if is_mukai and c == 2:
            ts += wind_effect * 1.5
        if is_oi_or_yoko and c in (3, 4):
            ts += wind_effect * 1.2
        tenkai_scores.append({'course': c, 'boat_number': s['boat']['boat_number'], 'ts': ts})

    tenkai_scores.sort(key=lambda x: x['ts'], reverse=True)
    top_tenkai = tenkai_scores[:4]

    # 裏熊×3: 展開スコア上位4艇の3連単コンボ（スコア上位3点）
    import itertools
    ura_combos = []
    seen_ura = set()
    top_bns = [t['boat_number'] for t in top_tenkai]
    for perm in itertools.permutations(top_bns, 3):
        combo_str = f"{perm[0]}-{perm[1]}-{perm[2]}"
        if combo_str in seen_ura:
            continue
        seen_ura.add(combo_str)
        # スコア = 1着TS×3 + 2着TS×2 + 3着TS
        ts_map = {t['boat_number']: t['ts'] for t in top_tenkai}
        combined = ts_map[perm[0]] * 3 + ts_map[perm[1]] * 2 + ts_map[perm[2]]
        ura_combos.append({'combo': combo_str, 'combined': combined})

    ura_combos.sort(key=lambda x: x['combined'], reverse=True)
    ura_3 = [{'combo': u['combo'], 'type': '裏熊'} for u in ura_combos[:3]]

    # 裏熊×1: 差し/まくり判断で1号艇を2着or3着に配置
    # まくり要因: 追い風/横風 + 3-4コースが上位スコア → 1号艇3着
    # 差し要因: 向かい風 + 2コース上位 → 1号艇2着
    outer_top3_courses = [s['course'] for s in outer_boats[:3]]
    makuri_score = 0
    sashi_score = 0
    if is_oi_or_yoko:
        makuri_score += wind_effect
    if is_mukai:
        sashi_score += wind_effect
    if 3 in outer_top3_courses or 4 in outer_top3_courses:
        makuri_score += 1.0
    if 2 in outer_top3_courses:
        sashi_score += 1.0

    # 1号艇の配置決定
    boat1_bn = None
    for s in scores:
        if s['course'] == 1:
            boat1_bn = s['boat']['boat_number']
            break
    if boat1_bn is None:
        boat1_bn = 1

    # 最も展開スコアが高い非1号艇を1着に
    best_first = top_tenkai[0]['boat_number'] if top_tenkai else 2
    # 2着候補（1着・1号艇以外）
    seconds = [t['boat_number'] for t in top_tenkai if t['boat_number'] not in (best_first, boat1_bn)]

    if makuri_score >= sashi_score:
        # まくり: 1号艇3着
        second_bn = seconds[0] if seconds else (top_tenkai[1]['boat_number'] if len(top_tenkai) > 1 else 2)
        ura_1_combo = f"{best_first}-{second_bn}-{boat1_bn}"
    else:
        # 差し: 1号艇2着
        third_bn = seconds[0] if seconds else (top_tenkai[1]['boat_number'] if len(top_tenkai) > 1 else 3)
        ura_1_combo = f"{best_first}-{boat1_bn}-{third_bn}"

    ura_1 = [{'combo': ura_1_combo, 'type': '裏熊'}]

    return manzoku + ura_3 + ura_1


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
    force_arekote = data.get('force_arekote', False)
    # 実戦はv4凍結中: USE_V5_LIVE/USE_V6_LIVEがTrueのときだけ各補正を使う
    # ただし裏熊モードは決まり手データ（主役判定・シナリオ）が必須なので常に渡す
    kimari_full_raw = data.get('kimari_full', None)
    kimari_full = kimari_full_raw if (USE_V5_LIVE or force_arekote) else None
    # extra_stats（超展開データ等）: 裏熊モードでは主役判定に使用、
    # 通常モードでも中穴予想のnakaana_score算出にのみ使う（本命/対抗のスコアには影響しない）
    extra_stats = data.get('extra_stats', None)
    result = predict(boats, kimari, venue=venue, wind=wind, nige_rate=nige_rate, force_arekote=force_arekote, kimari_full=kimari_full, hybrid_taikou=USE_V6_LIVE, v7_fix2nd=False, v7_third=USE_V7_THIRD_LIVE, extra_stats=extra_stats)
    return jsonify(result)


@app.route('/save_record', methods=['POST'])
def save_record():
    data = request.get_json()
    conn = get_db()
    conn.execute('''
        INSERT INTO records (race_date, venue, race_no, predictions, created_at, nige_rate, wind, input_data, model_version, is_womens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('race_date', ''),
        data.get('venue', ''),
        data.get('race_no', 0),
        json.dumps(data.get('predictions', []), ensure_ascii=False),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data.get('nige_rate'),
        data.get('wind'),
        json.dumps(data.get('input_data'), ensure_ascii=False) if data.get('input_data') else None,
        MODEL_VERSION,
        1 if data.get('is_womens') else 0
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

    # HTMLをファイルに保存（直接確認用）
    save_path = os.path.join(os.path.dirname(__file__), 'debug_result.html')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # 3連単セクション（2000文字）
    trio_idx = html.find('3連単')
    trio_snippet = html[trio_idx:trio_idx+2000].replace('\n', ' ') if trio_idx >= 0 else '(3連単 not found)'

    return jsonify({
        'url': url,
        'html_length': len(html),
        'boatcolor_snippets': snippets,
        'trio_section': trio_snippet,
        'saved_to': save_path,
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
        # 結果未確定か判断するヒントをHTMLから探す
        if 'boatColor' not in html:
            msg = 'レース結果がまだ公開されていません。終了後1〜3分待ってから再試行してください。'
        else:
            msg = '着順データが見つかりません。結果が反映されていないか、特殊なレース（中止・欠場等）の可能性があります。手動で入力してください。'
        return jsonify({'success': False, 'error': msg, 'url': url})

    r1, r2, r3 = boats_found[0], boats_found[1], boats_found[2]

    # --- 払戻金取得 ---
    # 実際のHTML構造: 3連単セクションにnumberSet1_number spanが並び、
    # その後の td に払戻金額が入っている
    # 例: <span class="numberSet1_number is-type1">1</span>...<span is-type6>6</span>...<span is-type2>2</span>
    # 金額は別の td に "2,340" 形式で入る

    # 実際の形式: <span class="is-payout1">&yen;420</span>
    # &yen; (¥記号のHTMLエンティティ) の後に金額が入っている
    m_payout = re.search(r'is-payout1[^>]*>\s*(?:&yen;|¥|￥)\s*([\d,]+)', html)
    if m_payout:
        try:
            payout = int(m_payout.group(1).replace(',', ''))
        except Exception:
            pass

    # フォールバック: &yen; を含む数字を3連単セクションから取得
    if payout == 0:
        for keyword in ['3連単', '三連単']:
            idx = html.find(keyword)
            if idx >= 0:
                area = html[idx:idx+2000]
                for a in re.findall(r'(?:&yen;|¥|￥)\s*([\d,]+)', area):
                    try:
                        val = int(a.replace(',', ''))
                        if val >= 100:
                            payout = val
                            break
                    except Exception:
                        pass
                if payout:
                    break

    # 最終フォールバック: is-payout / is-pay クラスから取得
    if payout == 0:
        for pat in [r'is-payout\d*[^>]*>\s*(?:&yen;|¥|￥)?\s*([\d,]+)', r'is-pay[^>]*>\s*(?:&yen;|¥|￥)?\s*([\d,]+)']:
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
    henkan = data.get('henkan') or None  # 返還艇（例: "4" や "4,5"）

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
        payout=?, purchase=?, is_hit=?, henkan=? WHERE id=?
    ''', (result_1st, result_2nd, result_3rd, payout, purchase, is_hit, henkan, record_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'is_hit': is_hit})


@app.route('/get_records')
def get_records():
    period = request.args.get('period', 'all')
    conn = get_db()
    if period == 'today':
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records WHERE race_date = date('now', '+9 hours') ORDER BY id DESC").fetchall()
    elif period == 'week':
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records WHERE race_date >= date('now', '-7 days') ORDER BY id DESC").fetchall()
    elif period == 'month':
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records WHERE race_date >= date('now', '-30 days') ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records ORDER BY id DESC").fetchall()
    conn.close()

    # tier判定用: 結果が出ている全レコードを会場別にまとめておく（期間フィルタと無関係に全期間で判定）
    conn2 = get_db()
    all_result_rows = conn2.execute(
        "SELECT venue, nige_rate, predictions, result_1st, result_2nd, result_3rd, payout, henkan FROM records WHERE result_1st IS NOT NULL"
    ).fetchall()
    conn2.close()
    venue_rows_by_venue = {}
    for rr in all_result_rows:
        venue_rows_by_venue.setdefault(rr['venue'], []).append(rr)
    tier_cache = {}

    records = []
    for r in rows:
        preds = json.loads(r['predictions'])
        honmei = next((p for p in preds if p.get('type') == '本命'), None)
        honmei_odds = honmei.get('odds') if honmei else None
        cache_key = (r['venue'], r['nige_rate'], honmei_odds)
        if cache_key not in tier_cache:
            tier_cache[cache_key] = _judge_tier(r['venue'], r['nige_rate'], honmei_odds, venue_rows_by_venue)
        records.append({
            'id': r['id'],
            'race_date': r['race_date'],
            'venue': r['venue'],
            'race_no': r['race_no'],
            'predictions': preds,
            'result_1st': r['result_1st'],
            'result_2nd': r['result_2nd'],
            'result_3rd': r['result_3rd'],
            'payout': r['payout'],
            'purchase': r['purchase'],
            'is_hit': r['is_hit'],
            'created_at': r['created_at'],
            'nige_rate': r['nige_rate'],
            'wind': r['wind'],
            'henkan': r['henkan'] if 'henkan' in r.keys() else None,
            'is_womens': bool(r['is_womens']) if 'is_womens' in r.keys() and r['is_womens'] is not None else False,
            'tier': tier_cache[cache_key],
        })
    return jsonify(records)


@app.route('/stats_summary', methods=['POST'])
def stats_summary():
    """会場×イン逃げ率帯の過去データをSQLiteで集計して返す"""
    data = request.get_json()
    venue = data.get('venue', '')
    nige_rate = data.get('nige_rate', None)

    conn = get_db()

    # 会場フィルタ
    if not venue:
        conn.close()
        return jsonify({'error': '会場未指定'})

    rows = conn.execute(
        "SELECT predictions, is_hit, payout, purchase, nige_rate, wind, result_1st, result_2nd, result_3rd, race_date FROM records WHERE venue=? AND purchase IS NOT NULL",
        (venue,)
    ).fetchall()
    conn.close()

    # イン逃げ率帯フィルタ（5%単位切り捨てを下限にする。ただし65%以上は一律65%以上でまとめる）
    if nige_rate is not None:
        nige_min = 65 if nige_rate >= 65 else math.floor(nige_rate / 5) * 5
        filtered = [r for r in rows if r['nige_rate'] is not None and r['nige_rate'] >= nige_min]
    else:
        nige_min = None
        filtered = list(rows)

    total = len(filtered)
    if total == 0:
        return jsonify({'total': 0, 'venue': venue, 'nige_rate': nige_rate, 'nige_min': nige_min})

    # 集計
    hits = sum(1 for r in filtered if r['is_hit'])
    total_purchase = sum(r['purchase'] or 0 for r in filtered)
    total_payout = sum((r['payout'] or 0) for r in filtered if r['is_hit'])
    hit_rate = round(hits / total * 100, 1)
    recovery = round(total_payout / total_purchase * 100, 1) if total_purchase > 0 else 0

    # 予想タイプ別集計（レース単位：そのレースでそのタイプが1点でも当たったか）
    type_stats = {}
    for r in filtered:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        has_result = r['result_1st'] and r['result_2nd'] and r['result_3rd']
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}" if has_result else None
        # このレースに含まれるタイプを収集
        types_in_race = {}
        for p in preds:
            t = p.get('type', '')
            if not t:
                continue
            if t not in types_in_race:
                types_in_race[t] = False
            if result_combo and p.get('combo') == result_combo:
                types_in_race[t] = True
        # レース単位でカウント
        for t, hit in types_in_race.items():
            if t not in type_stats:
                type_stats[t] = {'count': 0, 'hit': 0}
            type_stats[t]['count'] += 1
            if hit:
                type_stats[t]['hit'] += 1

    type_hit_rates = {}
    for t, s in type_stats.items():
        if s['count'] > 0:
            type_hit_rates[t] = {
                'count': s['count'],
                'hit': s['hit'],
                'rate': round(s['hit'] / s['count'] * 100, 1)
            }

    # 風向き別集計（is_hitのレースのみカウント）
    wind_stats = {}
    for r in filtered:
        w = r['wind'] if 'wind' in r.keys() else None
        if not w:
            w = '無風'
        if w not in wind_stats:
            wind_stats[w] = {'count': 0, 'hit': 0}
        wind_stats[w]['count'] += 1
        if r['is_hit']:
            wind_stats[w]['hit'] += 1

    # 風向き表示順（追い風→向かい風→横風→無風）
    WIND_ORDER = [
        '追い風1m', '追い風2m', '追い風3m以上',
        '向かい風1m', '向かい風2m', '向かい風3m以上',
        '横風（左）1m', '横風（左）2m', '横風（左）3m以上',
        '横風（右）1m', '横風（右）2m', '横風（右）3m以上',
        '無風', 'その他',
    ]
    wind_hit_rates = []
    seen_winds = set()
    for w in WIND_ORDER:
        if w in wind_stats:
            s = wind_stats[w]
            wind_hit_rates.append({
                'wind': w,
                'count': s['count'],
                'hit': s['hit'],
                'rate': round(s['hit'] / s['count'] * 100, 1) if s['count'] > 0 else 0,
            })
            seen_winds.add(w)
    # WIND_ORDERにないものも末尾に追加
    for w, s in wind_stats.items():
        if w not in seen_winds:
            wind_hit_rates.append({
                'wind': w,
                'count': s['count'],
                'hit': s['hit'],
                'rate': round(s['hit'] / s['count'] * 100, 1) if s['count'] > 0 else 0,
            })

    # 平均収支（1レースあたり）
    avg_profit = round((total_payout - total_purchase) / total, 0) if total > 0 else 0

    # 的中した時の平均払戻
    hit_records = [r for r in filtered if r['is_hit'] and r['payout']]
    avg_payout_on_hit = round(sum(r['payout'] for r in hit_records) / len(hit_records), 0) if hit_records else 0

    # 直近5戦の結果（新しい順）
    recent5 = []
    sorted_filtered = sorted(filtered, key=lambda r: r['race_date'] if 'race_date' in r.keys() else '', reverse=True)
    for r in sorted_filtered[:5]:
        recent5.append('hit' if r['is_hit'] else 'miss')

    # 最後に的中した日付
    last_hit_date = None
    for r in sorted_filtered:
        if r['is_hit']:
            last_hit_date = r['race_date'] if 'race_date' in r.keys() else None
            break

    # 1コース以外が1着になった率（荒れ率）
    upset_count = 0
    result_total = 0
    for r in filtered:
        if r['result_1st']:
            result_total += 1
            if str(r['result_1st']) != '1':
                upset_count += 1
    upset_rate = round(upset_count / result_total * 100, 1) if result_total > 0 else None

    # 推奨度判定
    if total < 5:
        level = 'unknown'
    elif recovery >= 100 and hit_rate >= 25:
        level = 'hot'
    elif recovery >= 90 and hit_rate >= 20:
        level = 'watch'
    else:
        level = 'pass'

    return jsonify({
        'venue': venue,
        'nige_rate': nige_rate,
        'nige_min': nige_min,
        'total': total,
        'hits': hits,
        'hit_rate': hit_rate,
        'recovery': recovery,
        'avg_profit': avg_profit,
        'avg_payout_on_hit': avg_payout_on_hit,
        'recent5': recent5,
        'last_hit_date': last_hit_date,
        'upset_rate': upset_rate,
        'type_hit_rates': type_hit_rates,
        'wind_hit_rates': wind_hit_rates,
        'level': level,
    })


@app.route('/payout_stats', methods=['GET'])
def payout_stats():
    """種別（本命/対抗/中穴/万舟）ごとの的中時平均配当を集計"""
    conn = get_db()
    rows = conn.execute(
        "SELECT predictions, result_1st, result_2nd, result_3rd, payout FROM records WHERE payout IS NOT NULL AND result_1st IS NOT NULL"
    ).fetchall()
    conn.close()

    TYPES = ['本命', '対抗', '中穴', '万舟']
    stats = {t: {'hits': 0, 'total': 0, 'payouts': []} for t in TYPES}

    for row in rows:
        try:
            preds = json.loads(row['predictions'])
        except Exception:
            continue
        result = f"{row['result_1st']}-{row['result_2nd']}-{row['result_3rd']}"
        # レース単位で集計（種別ごとに1レース=1カウント）
        types_in_race = {}
        for p in preds:
            t = p.get('type')
            if t not in TYPES:
                continue
            if t not in types_in_race:
                types_in_race[t] = False
            if p.get('combo') == result:
                types_in_race[t] = True
        for t, is_hit in types_in_race.items():
            stats[t]['total'] += 1
            if is_hit:
                stats[t]['hits'] += 1
                if row['payout'] and row['payout'] > 0:
                    stats[t]['payouts'].append(row['payout'])

    result_data = {}
    for t in TYPES:
        s = stats[t]
        avg_pay = round(sum(s['payouts']) / len(s['payouts'])) if s['payouts'] else None
        hit_rate = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        result_data[t] = {
            'hits': s['hits'],
            'total': s['total'],
            'hit_rate': hit_rate,
            'avg_payout': avg_pay,
        }

    return jsonify(result_data)


@app.route('/history_stats', methods=['POST'])
def history_stats():
    """絞り込み条件をサーバーで処理して集計結果を返す"""
    data = request.get_json()
    period     = data.get('period', 'all')
    venues     = data.get('venues', [])
    nige_min   = data.get('nige_min', 0)
    nige_max   = data.get('nige_max', 100)
    manzoku    = data.get('manzoku', False)
    kind       = data.get('kind', 'all')  # all / normal / ura（裏熊予想の分離）
    womens     = data.get('womens', 'all')  # all / only / exclude（女子戦の絞り込み）

    conn = get_db()
    if period == 'today':
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records WHERE race_date = date('now', '+9 hours') ORDER BY id DESC").fetchall()
    elif period == 'week':
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records WHERE race_date >= date('now', '-7 days') ORDER BY id DESC").fetchall()
    elif period == 'month':
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records WHERE race_date >= date('now', '-30 days') ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT id, race_date, venue, race_no, predictions, result_1st, result_2nd, result_3rd, payout, purchase, is_hit, created_at, nige_rate, wind, henkan, is_womens FROM records ORDER BY id DESC").fetchall()
    conn.close()

    # フィルター適用
    filtered = []
    for r in rows:
        if venues and r['venue'] not in venues:
            continue
        if nige_min > 0 or nige_max < 100:
            if r['nige_rate'] is None:
                continue
            if r['nige_rate'] < nige_min or r['nige_rate'] > nige_max:
                continue
        if manzoku and (not r['payout'] or r['payout'] < 10000):
            continue
        # 予想種別フィルター: 裏熊タイプを含むレコードか否かで判定
        if kind != 'all':
            is_ura = '"裏熊"' in (r['predictions'] or '')
            if kind == 'ura' and not is_ura:
                continue
            if kind == 'normal' and is_ura:
                continue
        if womens != 'all':
            is_w = bool(r['is_womens']) if 'is_womens' in r.keys() and r['is_womens'] is not None else False
            if womens == 'only' and not is_w:
                continue
            if womens == 'exclude' and is_w:
                continue
        filtered.append(r)

    total = len(filtered)
    with_result = [r for r in filtered if r['result_1st']]
    hits = [r for r in with_result if r['is_hit']]
    hit_rate = round(len(hits) / len(with_result) * 100, 1) if with_result else 0
    total_payout = sum(r['payout'] or 0 for r in hits)
    total_purchase = sum(r['purchase'] or 0 for r in with_result if r['purchase'])
    recovery = round(total_payout / total_purchase * 100, 1) if total_purchase > 0 else 0
    profit = total_payout - total_purchase

    # 種別別集計
    TYPES = ['本命', '対抗', '中穴', '万舟']
    TYPE_BY_INDEX = ['本命','本命','対抗','対抗','対抗','万舟','万舟']
    type_stats = {t: {'hits': 0, 'total': 0, 'payouts': []} for t in TYPES}
    for r in with_result:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        henkan_str = r['henkan'] if 'henkan' in r.keys() else None
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        types_in_race = {}
        for idx, p in enumerate(preds):
            t = p.get('type') or (TYPE_BY_INDEX[idx] if idx < len(TYPE_BY_INDEX) else '')
            if t not in TYPES:
                continue
            # 返還コンボは無効（外れでも的中でもない）
            if p.get('combo') and is_henkan_combo(p['combo'], henkan_str):
                continue
            if t not in types_in_race:
                types_in_race[t] = False
            if p.get('combo') == result_combo:
                types_in_race[t] = True
        for t, is_hit in types_in_race.items():
            type_stats[t]['total'] += 1
            if is_hit:
                type_stats[t]['hits'] += 1
                if r['payout'] and r['payout'] > 0:
                    type_stats[t]['payouts'].append(r['payout'])

    pattern = {}
    for t in TYPES:
        s = type_stats[t]
        avg_pay = round(sum(s['payouts']) / len(s['payouts'])) if s['payouts'] else None
        hr = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        breakeven = round(avg_pay * s['hits'] / s['total']) if avg_pay and s['total'] > 0 else None
        pattern[t] = {'hits': s['hits'], 'total': s['total'], 'hit_rate': hr,
                      'avg_payout': avg_pay, 'breakeven': breakeven}

    # 本命+対抗5点ベースの回収率（万舟のまぐれ当たりを除外した実力ベース）
    # 会場の得意/苦手判定に使用。100円/点で本命対抗のみ買った想定
    recovery_base, base_races = _calc_recovery_base(with_result)

    # 外れ方内訳（1着は合っていたか？）本命・対抗それぞれ集計
    def calc_miss_breakdown(target_type):
        mb = {'hit': 0, 'first_second': 0, 'first_only': 0, 'first_wrong': 0, 'total': 0}
        for r in with_result:
            try:
                preds = json.loads(r['predictions'])
            except Exception:
                continue
            henkan_str = r['henkan'] if 'henkan' in r.keys() else None
            combos = []
            for idx, p in enumerate(preds):
                t = p.get('type') or (TYPE_BY_INDEX[idx] if idx < len(TYPE_BY_INDEX) else '')
                if t == target_type and p.get('combo') and not is_henkan_combo(p['combo'], henkan_str):
                    combos.append(p['combo'])
            if not combos:
                continue
            mb['total'] += 1
            result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
            if result_combo in combos:
                mb['hit'] += 1
                continue
            first_match = any(c.split('-')[0] == str(r['result_1st']) for c in combos)
            second_match = any(c.split('-')[0] == str(r['result_1st']) and c.split('-')[1] == str(r['result_2nd']) for c in combos)
            if second_match:
                mb['first_second'] += 1
            elif first_match:
                mb['first_only'] += 1
            else:
                mb['first_wrong'] += 1
        return mb

    miss_breakdown = calc_miss_breakdown('本命')
    miss_breakdown_taikou = calc_miss_breakdown('対抗')

    # 会場別集計
    venue_stats = {}
    for r in with_result:
        v = r['venue']
        if v not in venue_stats:
            venue_stats[v] = {'hits': 0, 'total': 0, 'payout': 0, 'purchase': 0}
        venue_stats[v]['total'] += 1
        if r['is_hit']:
            venue_stats[v]['hits'] += 1
            venue_stats[v]['payout'] += r['payout'] or 0
        venue_stats[v]['purchase'] += r['purchase'] or 0

    venue_list = []
    for v, s in venue_stats.items():
        hr = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        rec = round(s['payout'] / s['purchase'] * 100, 1) if s['purchase'] > 0 else 0
        venue_list.append({'venue': v, 'hits': s['hits'], 'total': s['total'],
                           'hit_rate': hr, 'recovery': rec})
    venue_list.sort(key=lambda x: x['hit_rate'], reverse=True)

    # 収支推移（日別）
    daily = {}
    for r in sorted(with_result, key=lambda x: x['race_date']):
        d = r['race_date']
        if d not in daily:
            daily[d] = {'payout': 0, 'purchase': 0}
        daily[d]['payout'] += r['payout'] or 0
        daily[d]['purchase'] += r['purchase'] or 0
    profit_series = []
    cumulative = 0
    for d, s in sorted(daily.items()):
        cumulative += s['payout'] - s['purchase']
        profit_series.append({'date': d, 'cumulative': cumulative})

    return jsonify({
        'total': total,
        'with_result': len(with_result),
        'hits': len(hits),
        'hit_rate': hit_rate,
        'recovery': recovery,
        'profit': profit,
        'total_payout': total_payout,
        'total_purchase': total_purchase,
        'pattern': pattern,
        'recovery_base': recovery_base,
        'base_races': base_races,
        'miss_breakdown': miss_breakdown,
        'miss_breakdown_taikou': miss_breakdown_taikou,
        'venue_list': venue_list,
        'profit_series': profit_series,
    })


@app.route('/backtest', methods=['POST'])
def backtest():
    """入力データが保存されたレースを現行ロジックで再予想し、保存時の予想と比較。
    成績記録の絞り込み（期間・会場・イン逃げ率）と連動する"""
    data = request.get_json() or {}
    period = data.get('period', 'all')
    venues = data.get('venues', [])
    nige_min = data.get('nige_min', 0)
    nige_max = data.get('nige_max', 100)
    honmei_odds_min = data.get('honmei_odds_min', 0)  # 本命1点目オッズの下限（勝負レース相当の絞り込み用）
    # 検証モード: どの候補ロジックを乗せて再予想するか（v6=実戦同等）
    variant = data.get('variant', 'v7b')
    VARIANTS = {
        'v6':      {'v5': False, 'fix2nd': False, 'third': False, 'label': 'v6のみ（旧・実戦相当）'},
        'v5only':  {'v5': True,  'fix2nd': False, 'third': False, 'label': 'v6+v5のみ（決まり手展開補正・不採用）'},
        'v7a':     {'v5': False, 'fix2nd': True,  'third': False, 'label': 'v6+2着バグ修正のみ（不採用）'},
        'v7b':     {'v5': False, 'fix2nd': False, 'third': True,  'label': 'v6+3着差し替えのみ = v7（現在の実戦）'},
        'v7':      {'v5': False, 'fix2nd': True,  'third': True,  'label': 'v6+v7フル（2着修正込み・不採用）'},
        'full':    {'v5': True,  'fix2nd': True,  'third': True,  'label': 'v5+v7全部入り（不採用要素込み）'},
        'groupA':  {'v5': False, 'fix2nd': False, 'third': False, 'label': 'グループA単独（当日の調子: 展示タイム/ST/チルト/周回直線まわり足）'},
        'groupB':  {'v5': False, 'fix2nd': False, 'third': False, 'label': 'グループB単独（地力・機材: 選手勝率/級別/モーター/会場コース別）'},
    }
    is_group_variant = variant in ('groupA', 'groupB')
    vconf = VARIANTS.get(variant, VARIANTS['full'])
    use_v5 = vconf['v5']
    use_fix2nd = vconf['fix2nd']
    use_third = vconf['third']

    conn = get_db()
    q = "SELECT * FROM records WHERE input_data IS NOT NULL AND result_1st IS NOT NULL"
    if period == 'today':
        q += " AND race_date = date('now', '+9 hours')"
    elif period == 'week':
        q += " AND race_date >= date('now', '-7 days')"
    elif period == 'month':
        q += " AND race_date >= date('now', '-30 days')"
    rows = conn.execute(q + " ORDER BY id").fetchall()
    conn.close()

    # 会場・イン逃げ率・本命オッズフィルター
    filtered_rows = []
    for r in rows:
        if venues and r['venue'] not in venues:
            continue
        if nige_min > 0 or nige_max < 100:
            if r['nige_rate'] is None:
                continue
            if r['nige_rate'] < nige_min or r['nige_rate'] > nige_max:
                continue
        if honmei_odds_min > 0:
            try:
                preds = json.loads(r['predictions'])
            except Exception:
                continue
            h1 = next((p for p in preds if p.get('type') == '本命'), None)
            if not h1 or h1.get('odds') is None or h1['odds'] < honmei_odds_min:
                continue
        filtered_rows.append(r)
    rows = filtered_rows

    def eval_preds(preds, result_combo):
        """予想リストの種別ごとの的中を判定（レースベース）"""
        out = {}
        for p in preds:
            t = p.get('type')
            if not t:
                continue
            if t not in out:
                out[t] = False
            if p.get('combo') == result_combo:
                out[t] = True
        return out

    TYPES = ['本命', '対抗', '中穴', '万舟']
    stored_stats = {t: {'hits': 0, 'total': 0} for t in TYPES}
    new_stats = {t: {'hits': 0, 'total': 0} for t in TYPES}
    n_races = 0
    # v3対抗シミュレーション（旧ロジック: 1位艇1着固定でスコア順3〜5番目）
    v3_taikou = {'hits': 0, 'total': 0}
    # 回収率比較（odds_allが保存されているレースのみ、本命+対抗5点ベース）
    ev_races = 0
    stored_purchase = stored_payout = 0
    new_purchase = new_payout = 0
    v3_purchase = v3_payout = 0

    for r in rows:
        try:
            inp = json.loads(r['input_data'])
            stored_preds = json.loads(r['predictions'])
        except Exception:
            continue
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"

        # 現行ロジックで再予想（variantで検証対象を切り替え）
        try:
            if is_group_variant:
                # 単一要素モデル比較: グループA/Bのみのスコアで再予想（v6/v7とは別の簡易ロジック）
                new_result = predict_single_factor(inp.get('boats', []), venue=inp.get('venue'),
                                                    group='A' if variant == 'groupA' else 'B')
            else:
                new_result = predict(
                    inp.get('boats', []),
                    kimari=inp.get('kimari'),
                    venue=inp.get('venue'),
                    wind=inp.get('wind'),
                    nige_rate=inp.get('nige_rate'),
                    kimari_full=inp.get('kimari_full') if use_v5 else None,
                    hybrid_taikou=True,  # v6ハイブリッド対抗（採用済み）は常にON
                    v7_fix2nd=use_fix2nd,
                    v7_third=use_third,
                )
            new_preds = new_result['predictions']
        except Exception:
            continue

        n_races += 1
        for t, is_hit in eval_preds(stored_preds, result_combo).items():
            if t in stored_stats:
                stored_stats[t]['total'] += 1
                if is_hit:
                    stored_stats[t]['hits'] += 1
        for t, is_hit in eval_preds(new_preds, result_combo).items():
            if t in new_stats:
                new_stats[t]['total'] += 1
                if is_hit:
                    new_stats[t]['hits'] += 1

        # v3対抗シミュ: 1位艇1着固定のスコア順コンボ3〜5番目（旧対抗）
        v3_combos = []
        try:
            top_bn = new_result['score_order'][0]['boat_number']
            top_first = [c['combo'] for c in new_result['candidates']
                         if c['combo'].startswith(f"{top_bn}-")]
            v3_combos = top_first[2:5]  # 1〜2番目は本命相当なので3〜5番目が旧対抗
        except Exception:
            pass
        if v3_combos:
            v3_taikou['total'] += 1
            if result_combo in v3_combos:
                v3_taikou['hits'] += 1

        # 回収率比較: odds_allがあるレースのみ、本命+対抗の5点を100円ずつ買った想定
        odds_all = inp.get('odds_all')
        if odds_all:
            result_odds = safe_float(odds_all.get(result_combo, 0))
            stored_base = [p for p in stored_preds if p.get('type') in ('本命', '対抗')]
            new_base = [p for p in new_preds if p.get('type') in ('本命', '対抗')]
            if stored_base and new_base:
                ev_races += 1
                stored_purchase += 100 * len(stored_base)
                new_purchase += 100 * len(new_base)
                if any(p.get('combo') == result_combo for p in stored_base):
                    stored_payout += int(result_odds * 100)
                if any(p.get('combo') == result_combo for p in new_base):
                    new_payout += int(result_odds * 100)
                # v3の5点 = 本命2点（現行と同じ）+ v3対抗3点
                if v3_combos:
                    honmei_combos = [p.get('combo') for p in new_preds if p.get('type') == '本命']
                    v3_set = honmei_combos + v3_combos
                    v3_purchase += 100 * len(v3_set)
                    if result_combo in v3_set:
                        v3_payout += int(result_odds * 100)

    def fmt(stats):
        out = {}
        for t in TYPES:
            s = stats[t]
            out[t] = {'hits': s['hits'], 'total': s['total'],
                      'hit_rate': round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0}
        return out

    recovery = None
    if ev_races > 0 and stored_purchase > 0 and new_purchase > 0:
        recovery = {
            'races': ev_races,
            'stored': round(stored_payout / stored_purchase * 100, 1),
            'current': round(new_payout / new_purchase * 100, 1),
            'stored_payout': stored_payout,
            'current_payout': new_payout,
        }
        if v3_purchase > 0:
            recovery['v3'] = round(v3_payout / v3_purchase * 100, 1)
            recovery['v3_payout'] = v3_payout

    v3_result = None
    if v3_taikou['total'] > 0:
        v3_result = {'hits': v3_taikou['hits'], 'total': v3_taikou['total'],
                     'hit_rate': round(v3_taikou['hits'] / v3_taikou['total'] * 100, 1)}

    return jsonify({
        'races': n_races,
        'model_version': vconf['label'] + '（実戦はv6）',
        'stored': fmt(stored_stats),
        'current': fmt(new_stats),
        'v3_taikou': v3_result,
        'recovery': recovery,
    })


def _filtered_result_rows(data):
    """成績記録の絞り込み条件（期間・会場・逃げ率）で結果入力済みレコードを取得する共通処理"""
    period = data.get('period', 'all')
    venues = data.get('venues', [])
    nige_min = data.get('nige_min', 0)
    nige_max = data.get('nige_max', 100)
    womens = data.get('womens', 'all')
    conn = get_db()
    q = "SELECT * FROM records WHERE result_1st IS NOT NULL"
    if period == 'today':
        q += " AND race_date = date('now', '+9 hours')"
    elif period == 'week':
        q += " AND race_date >= date('now', '-7 days')"
    elif period == 'month':
        q += " AND race_date >= date('now', '-30 days')"
    rows = conn.execute(q + " ORDER BY id").fetchall()
    conn.close()
    out = []
    for r in rows:
        if venues and r['venue'] not in venues:
            continue
        if nige_min > 0 or nige_max < 100:
            if r['nige_rate'] is None:
                continue
            if r['nige_rate'] < nige_min or r['nige_rate'] > nige_max:
                continue
        if womens != 'all':
            is_w = bool(r['is_womens']) if 'is_womens' in r.keys() and r['is_womens'] is not None else False
            if womens == 'only' and not is_w:
                continue
            if womens == 'exclude' and is_w:
                continue
        out.append(r)
    return out


@app.route('/nirentan_check', methods=['POST'])
def nirentan_check():
    """2連単換算チェック: 本命/対抗の各コンボを1-2着だけに切り詰めて、
    もし2連単で買っていたら的中していたかを過去データから集計する。
    bi_odds（2連単オッズ）が保存されているレースは回収率も併せて計算する。
    """
    data = request.get_json() or {}
    rows = _filtered_result_rows(data)

    TYPE_BY_INDEX = ['本命', '本命', '対抗', '対抗', '対抗', '万舟', '万舟']
    stats = {
        '本命': {'hits': 0, 'total': 0, 'purchase': 0, 'payout': 0, 'has_odds_races': 0},
        '対抗': {'hits': 0, 'total': 0, 'purchase': 0, 'payout': 0, 'has_odds_races': 0},
        '裏熊': {'hits': 0, 'total': 0, 'purchase': 0, 'payout': 0, 'has_odds_races': 0},
    }

    for r in rows:
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        try:
            inp = json.loads(r['input_data']) if r['input_data'] else None
        except Exception:
            inp = None
        bi_odds = (inp or {}).get('bi_odds')
        result_ni = f"{r['result_1st']}-{r['result_2nd']}"

        henkan_str = r['henkan'] if 'henkan' in r.keys() else None
        henkan_boats = set((henkan_str or '').split(',')) if henkan_str else set()

        for t in ('本命', '対抗', '裏熊'):
            combos = []
            for idx, p in enumerate(preds):
                pt = p.get('type') or (TYPE_BY_INDEX[idx] if idx < len(TYPE_BY_INDEX) else '')
                combo = p.get('combo')
                if pt != t or not combo:
                    continue
                parts = combo.split('-')
                if len(parts) < 2 or (henkan_boats & set(parts)):
                    continue
                ni = f"{parts[0]}-{parts[1]}"
                if ni not in combos:
                    combos.append(ni)
            if not combos:
                continue
            s = stats[t]
            s['total'] += 1
            is_hit = result_ni in combos
            if is_hit:
                s['hits'] += 1
            if bi_odds:
                s['has_odds_races'] += 1
                s['purchase'] += 100 * len(combos)
                if is_hit:
                    s['payout'] += int(safe_float(bi_odds.get(result_ni, 0)) * 100)

    out = {}
    for t, s in stats.items():
        hr = round(s['hits'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        rec = round(s['payout'] / s['purchase'] * 100, 1) if s['purchase'] > 0 else None
        out[t] = {'hits': s['hits'], 'total': s['total'], 'hit_rate': hr,
                  'recovery': rec, 'odds_races': s['has_odds_races']}
    return jsonify(out)


@app.route('/ura_backtest', methods=['POST'])
def ura_backtest():
    """裏熊バックテスト: 裏熊レコードを現行の裏熊ロジックで再予想し、
    出動条件クリア数別の成績と保存時予想との比較を出す"""
    data = request.get_json() or {}
    rows = _filtered_result_rows(data)

    stored_stats = {'hits': 0, 'races': 0, 'purchase': 0, 'payout': 0}
    current_stats = {'hits': 0, 'races': 0, 'purchase': 0, 'payout': 0}
    # 出動条件クリア数別（現行ロジックで判定し直した値）
    by_clear = {}

    for r in rows:
        if '"裏熊"' not in (r['predictions'] or ''):
            continue
        try:
            inp = json.loads(r['input_data']) if r['input_data'] else None
            stored_preds = json.loads(r['predictions'])
        except Exception:
            continue
        if not inp:
            continue
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        odds_all = inp.get('odds_all') or {}
        result_odds = safe_float(odds_all.get(result_combo, 0))

        # 保存時の裏熊予想の成績
        stored_combos = [p['combo'] for p in stored_preds if p.get('combo')]
        if stored_combos:
            stored_stats['races'] += 1
            stored_stats['purchase'] += 100 * len(stored_combos)
            if result_combo in stored_combos:
                stored_stats['hits'] += 1
                if r['payout']:
                    stored_stats['payout'] += r['payout']
                elif result_odds:
                    stored_stats['payout'] += int(result_odds * 100)

        # 現行ロジックで再予想
        try:
            pred = predict(
                inp.get('boats', []),
                kimari=inp.get('kimari'),
                venue=inp.get('venue'),
                wind=inp.get('wind'),
                nige_rate=inp.get('nige_rate'),
                force_arekote=True,
                kimari_full=inp.get('kimari_full'),
                extra_stats=inp.get('extra_stats'),
            )
        except Exception:
            continue
        ura = pred.get('ura_judge') or {}
        conds = ura.get('conds') or {}
        raw_combos = [p['combo'] for p in (pred.get('arekote_predictions') or [])]
        new_combos = raw_combos
        # オッズ足切り（20倍未満）を再現
        if odds_all:
            new_combos = [c for c in raw_combos if safe_float(odds_all.get(c, 999)) >= 20]
        # 実戦画面（index.html）はオッズ妙味を4つ目の出動条件として扱い「n/4」表示にしているため、
        # ここでも同じ基準でclear/totalを揃える（揃えないとバックテストの「n/3」と実戦の「n/4」がズレる）
        clear = conds.get('clear', 0)
        total = conds.get('total', 3)
        if odds_all:
            all_cut = len(raw_combos) > 0 and len(new_combos) == 0
            total += 1
            if not all_cut:
                clear += 1
        clear_label = f"{clear}/{total}"
        if new_combos:
            current_stats['races'] += 1
            current_stats['purchase'] += 100 * len(new_combos)
            hit = result_combo in new_combos
            if hit:
                current_stats['hits'] += 1
                if result_odds:
                    current_stats['payout'] += int(result_odds * 100)
            if clear_label not in by_clear:
                by_clear[clear_label] = {'hits': 0, 'races': 0, 'purchase': 0, 'payout': 0}
            b = by_clear[clear_label]
            b['races'] += 1
            b['purchase'] += 100 * len(new_combos)
            if hit:
                b['hits'] += 1
                if result_odds:
                    b['payout'] += int(result_odds * 100)

    def fmt(s):
        return {
            'races': s['races'], 'hits': s['hits'],
            'hit_rate': round(s['hits'] / s['races'] * 100, 1) if s['races'] > 0 else 0,
            'purchase': s['purchase'], 'payout': s['payout'],
            'recovery': round(s['payout'] / s['purchase'] * 100, 1) if s['purchase'] > 0 else 0,
        }

    return jsonify({
        'stored': fmt(stored_stats),
        'current': fmt(current_stats),
        'by_clear': {k: fmt(v) for k, v in sorted(by_clear.items(), key=lambda kv: tuple(map(int, kv[0].split('/'))), reverse=True)},
    })


@app.route('/strategy_sim', methods=['POST'])
def strategy_sim():
    """買い方シミュレーター: odds_all付きレースで複数の買い方の回収率を一括比較"""
    data = request.get_json() or {}
    rows = _filtered_result_rows(data)

    # 通常予想向け戦略
    strategies = {
        'full5':    {'name': '5点フル（本命2+対抗3）', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'honmei2':  {'name': '本命2点のみ', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'skip1':    {'name': '本命1点目抜き4点', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'taikou3':  {'name': '対抗3点のみ', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'full7':    {'name': '7点（5点+万舟2点）', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'smart':    {'name': 'スマート（本命7倍未満は本命2点のみ、他は5点）', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'ruleA':    {'name': 'A: 本命どちらかが7倍未満なら見送り', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'ruleB':    {'name': 'B: 安い方の本命1点だけ除外し4点', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'ruleC':    {'name': 'C: 本命どちらかが7倍未満なら対抗抜き本命2点のみ', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
    }
    # 裏熊向け戦略
    ura_strategies = {
        'ura_all':    {'name': '裏熊: 全点', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
        'ura_30_150': {'name': '裏熊: 30〜150倍のみ', 'purchase': 0, 'payout': 0, 'hits': 0, 'races': 0},
    }

    def settle(strat, combos, result_combo, result_odds):
        if not combos:
            return
        strat['races'] += 1
        strat['purchase'] += 100 * len(combos)
        if result_combo in combos:
            strat['payout'] += int(result_odds * 100)
            strat['hits'] += 1

    for r in rows:
        try:
            inp = json.loads(r['input_data']) if r['input_data'] else None
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        odds_all = (inp or {}).get('odds_all')
        if not odds_all:
            continue
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        result_odds = safe_float(odds_all.get(result_combo, 0))

        is_ura = any(p.get('type') == '裏熊' for p in preds)
        if is_ura:
            all_combos = [p['combo'] for p in preds if p.get('combo')]
            band = [c for c in all_combos if 30 <= safe_float(odds_all.get(c, 0)) <= 150]
            settle(ura_strategies['ura_all'], all_combos, result_combo, result_odds)
            settle(ura_strategies['ura_30_150'], band, result_combo, result_odds)
            continue

        honmei_preds = [p for p in preds if p.get('type') == '本命' and p.get('combo')]
        honmei = [p['combo'] for p in honmei_preds]
        taikou = [p['combo'] for p in preds if p.get('type') == '対抗' and p.get('combo')]
        manshu = [p['combo'] for p in preds if p.get('type') == '万舟' and p.get('combo')]
        if not honmei:
            continue
        base5 = honmei + taikou
        settle(strategies['full5'], base5, result_combo, result_odds)
        settle(strategies['honmei2'], honmei, result_combo, result_odds)
        settle(strategies['skip1'], honmei[1:] + taikou, result_combo, result_odds)
        settle(strategies['taikou3'], taikou, result_combo, result_odds)
        settle(strategies['full7'], base5 + manshu, result_combo, result_odds)
        # スマート: 本命1点目のオッズだけで判定（従来の運用パターン）
        h1_odds = honmei_preds[0].get('odds') if honmei_preds else None
        smart_set = honmei if (h1_odds is not None and h1_odds < 7) else base5
        settle(strategies['smart'], smart_set, result_combo, result_odds)

        # A/B/C: 本命2点のうち「最も安い方」のオッズで判定（オッズ未記録の点は判定対象外）
        honmei_odds_list = [p['odds'] for p in honmei_preds if p.get('odds') is not None]
        min_honmei_odds = min(honmei_odds_list) if honmei_odds_list else None
        is_cheap = min_honmei_odds is not None and min_honmei_odds < 7
        if min_honmei_odds is not None:
            # A: 安い本命があるレースはまるごと見送り（0点=settleされずrace集計外）
            if not is_cheap:
                settle(strategies['ruleA'], base5, result_combo, result_odds)
            # B: 安い方の本命1点だけ除外し、残り（もう1本命+対抗3）を買う
            if is_cheap:
                cheapest_combo = min(honmei_preds, key=lambda p: p.get('odds', 999))['combo']
                b_set = [c for c in honmei if c != cheapest_combo] + taikou
            else:
                b_set = base5
            settle(strategies['ruleB'], b_set, result_combo, result_odds)
            # C: 安い本命があれば対抗を切って本命2点のみ、なければ5点フル
            c_set = honmei if is_cheap else base5
            settle(strategies['ruleC'], c_set, result_combo, result_odds)

    def fmt(d):
        out = []
        for key, s in d.items():
            rec = round(s['payout'] / s['purchase'] * 100, 1) if s['purchase'] > 0 else 0
            hr = round(s['hits'] / s['races'] * 100, 1) if s['races'] > 0 else 0
            out.append({'key': key, 'name': s['name'], 'races': s['races'], 'hits': s['hits'],
                        'hit_rate': hr, 'purchase': s['purchase'], 'payout': s['payout'],
                        'recovery': rec, 'profit': s['payout'] - s['purchase']})
        return out

    return jsonify({'normal': fmt(strategies), 'ura': fmt(ura_strategies)})


@app.route('/ev_calibration', methods=['POST'])
def ev_calibration():
    """EVキャリブレーション: モデル確率×オッズ（EV）帯ごとに実際の回収率を検証"""
    data = request.get_json() or {}
    rows = _filtered_result_rows(data)

    BUCKETS = [
        {'key': 'ev_lt70',   'label': 'EV 70%未満',    'lo': 0.0, 'hi': 0.7},
        {'key': 'ev_70_100', 'label': 'EV 70〜100%',   'lo': 0.7, 'hi': 1.0},
        {'key': 'ev_100_130','label': 'EV 100〜130%',  'lo': 1.0, 'hi': 1.3},
        {'key': 'ev_ge130',  'label': 'EV 130%以上',   'lo': 1.3, 'hi': 9999.0},
    ]
    stats = {b['key']: {'label': b['label'], 'bets': 0, 'hits': 0, 'purchase': 0, 'payout': 0,
                        'prob_sum': 0.0} for b in BUCKETS}
    n_races = 0

    for r in rows:
        try:
            inp = json.loads(r['input_data']) if r['input_data'] else None
        except Exception:
            continue
        if not inp:
            continue
        odds_all = inp.get('odds_all')
        if not odds_all:
            continue
        try:
            pred_result = predict(
                inp.get('boats', []),
                kimari=inp.get('kimari'),
                venue=inp.get('venue'),
                wind=inp.get('wind'),
                nige_rate=inp.get('nige_rate'),
            )
        except Exception:
            continue
        n_races += 1
        result_combo = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        for c in pred_result.get('candidates', []):
            prob = c.get('prob')
            odds = safe_float(odds_all.get(c['combo'], 0))
            if prob is None or odds <= 0:
                continue
            ev = prob * odds
            for b in BUCKETS:
                if b['lo'] <= ev < b['hi']:
                    s = stats[b['key']]
                    s['bets'] += 1
                    s['purchase'] += 100
                    s['prob_sum'] += prob
                    if c['combo'] == result_combo:
                        s['hits'] += 1
                        s['payout'] += int(odds * 100)
                    break

    out = []
    for b in BUCKETS:
        s = stats[b['key']]
        rec = round(s['payout'] / s['purchase'] * 100, 1) if s['purchase'] > 0 else 0
        actual_hr = round(s['hits'] / s['bets'] * 100, 2) if s['bets'] > 0 else 0
        model_hr = round(s['prob_sum'] / s['bets'] * 100, 2) if s['bets'] > 0 else 0
        out.append({'label': s['label'], 'bets': s['bets'], 'hits': s['hits'],
                    'model_hit_rate': model_hr, 'actual_hit_rate': actual_hr,
                    'recovery': rec})
    return jsonify({'races': n_races, 'buckets': out})


@app.route('/weekly_report', methods=['POST'])
def weekly_report():
    """週次分析レポート: 直近の成績・勝負レース・会場昇格候補・検証状況をまとめる"""
    data = request.get_json() or {}
    if 'period' not in data:
        data['period'] = 'week'
    rows = _filtered_result_rows(data)

    def is_ura_row(r):
        return '"裏熊"' in (r['predictions'] or '')

    normal_rows = [r for r in rows if not is_ura_row(r)]
    ura_rows = [r for r in rows if is_ura_row(r)]

    def summary(rs):
        hits = [r for r in rs if r['is_hit']]
        payout = sum(r['payout'] or 0 for r in hits)
        purchase = sum(r['purchase'] or 0 for r in rs if r['purchase'] is not None)
        return {
            'races': len(rs), 'hits': len(hits),
            'hit_rate': round(len(hits) / len(rs) * 100, 1) if rs else 0,
            'payout': payout, 'purchase': purchase,
            'recovery': round(payout / purchase * 100, 1) if purchase > 0 else 0,
            'profit': payout - purchase,
        }

    # ✅勝負レース相当（得意会場+逃げ率65%+本命1点目7倍以上）
    shobu_rows = []
    for r in normal_rows:
        if r['nige_rate'] is None or r['nige_rate'] < 65 or r['venue'] not in TOKUI_VENUES:
            continue
        try:
            preds = json.loads(r['predictions'])
        except Exception:
            continue
        h1 = next((p for p in preds if p.get('type') == '本命'), None)
        if h1 and h1.get('odds') is not None and h1['odds'] >= 7:
            shobu_rows.append(r)

    # 検証中会場の状況（逃げ率65%以上のレースの実力回収率）
    kensho_status = []
    for v in KENSHO_VENUES:
        vrows = [r for r in normal_rows if r['venue'] == v and r['nige_rate'] is not None and r['nige_rate'] >= 65]
        if not vrows:
            continue
        purchase = 0
        payout = 0
        hits = 0
        for r in vrows:
            try:
                preds = json.loads(r['predictions'])
            except Exception:
                continue
            base = [p['combo'] for p in preds if p.get('type') in ('本命', '対抗') and p.get('combo')]
            if not base:
                continue
            purchase += 100 * len(base)
            rc = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
            if rc in base and r['payout']:
                payout += r['payout']
                hits += 1
        rec = round(payout / purchase * 100, 1) if purchase > 0 else 0
        kensho_status.append({
            'venue': v, 'races': len(vrows), 'hits': hits, 'recovery_base': rec,
            'need_more': max(0, 10 - len(vrows)),
            'verdict': '昇格候補' if (len(vrows) >= 10 and rec >= 100) else ('降格候補' if (len(vrows) >= 10 and rec < 80) else 'データ蓄積中'),
        })
    kensho_status.sort(key=lambda x: x['recovery_base'], reverse=True)

    # 女子戦別の成績（通常予想のみ対象。荒れやすいという仮説の検証用）
    womens_rows = [r for r in normal_rows if r['is_womens']]
    non_womens_rows = [r for r in normal_rows if not r['is_womens']]

    # v5再評価の進捗
    conn = get_db()
    input_data_count = conn.execute(
        "SELECT COUNT(*) FROM records WHERE input_data IS NOT NULL AND result_1st IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    return jsonify({
        'total': summary(rows),
        'normal': summary(normal_rows),
        'shobu': summary(shobu_rows),
        'ura': summary(ura_rows),
        'womens': summary(womens_rows),
        'non_womens': summary(non_womens_rows),
        'kensho_status': kensho_status,
        'input_data_count': input_data_count,
        'v5_reeval_at': 150,
    })


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
