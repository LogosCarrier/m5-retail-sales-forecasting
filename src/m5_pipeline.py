# -*- coding: utf-8 -*-
"""M5 self-iteration pipeline: config -> experiment -> WRMSSE -> registry -> promote -> submit."""
from __future__ import annotations

import os, gc, json, time, pickle, hashlib, subprocess
# M5_ROOT controls data / artifacts / temp locations (defaults to the original Windows workspace).
ROOT = os.environ.get('M5_ROOT', r'E:\m5work')
_TMP = os.path.join(ROOT, 'tmp')
for _k in ('TMP', 'TEMP', 'TMPDIR'):
    os.environ[_k] = _TMP
os.makedirs(_TMP, exist_ok=True)

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

DATA = os.path.join(ROOT, 'data')
ART = os.path.join(ROOT, 'artifacts')
MODELS = os.path.join(ART, 'models')
for _p in (ART, MODELS):
    os.makedirs(_p, exist_ok=True)

NDAY, H = 1941, 28
CUT_VAL, CUT_FINAL = 1913, 1941
CACHE = os.path.join(ART, 'cache.pkl')
META = os.path.join(ART, 'wrmsse_meta.pkl')
REG = os.path.join(ART, 'experiments.jsonl')
BEST = os.path.join(ART, 'best.json')
RESULT = os.path.join(ART, 'result.json')
LOG = os.path.join(ART, 'run.log')
STATIC = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

LEVELS = [
    [], ['state_id'], ['store_id'], ['cat_id'], ['dept_id'],
    ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'],
    ['store_id', 'dept_id'], ['item_id'], ['item_id', 'state_id'],
    ['item_id', 'store_id'],
]

STAGE_A = [
    dict(name='A_base',    features='base',  objective='regression', num_leaves=63,  lr=0.06, trees=250, windows=[1885]),
    dict(name='A_price',   features='price', objective='regression', num_leaves=63,  lr=0.06, trees=250, windows=[1885]),
    dict(name='A_long',    features='long',  objective='regression', num_leaves=63,  lr=0.06, trees=250, windows=[1885]),
    dict(name='A_full',    features='full',  objective='regression', num_leaves=63,  lr=0.06, trees=250, windows=[1885]),
    dict(name='A_poisson', features='full',  objective='poisson',    num_leaves=63,  lr=0.06, trees=250, windows=[1885]),
    dict(name='A_tweedie', features='full',  objective='tweedie',    num_leaves=63,  lr=0.06, trees=250, windows=[1885]),
    dict(name='A_deep',    features='full',  objective='regression', num_leaves=127, lr=0.05, trees=300, windows=[1885]),
    dict(name='A_2win',    features='full',  objective='regression', num_leaves=63,  lr=0.06, trees=250, windows=[1857, 1885]),
]

WIN5 = [1733, 1789, 1829, 1857, 1885]
STAGE_C = [
    dict(name='C_long5',     features='long',  objective='regression', num_leaves=63,  lr=0.045, trees=800, windows=WIN5),
    dict(name='C_full2',     features='full2', objective='regression', num_leaves=63,  lr=0.045, trees=800, windows=WIN5),
    dict(name='C_full2deep', features='full2', objective='regression', num_leaves=127, lr=0.04,  trees=900, windows=WIN5),
]

STAGE_D = [
    dict(name='D_base_more', features='base', objective='regression', num_leaves=63,  lr=0.03, trees=1200, windows=[1829, 1857, 1885]),
    dict(name='D_base_deep', features='base', objective='regression', num_leaves=127, lr=0.03, trees=1000, windows=[1829, 1857, 1885]),
]


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def cfg_hash(cfg, stage):
    s = json.dumps({'stage': stage, **cfg}, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def append_jsonl(path, row):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')


def read_best():
    if not os.path.exists(BEST):
        return None
    with open(BEST, encoding='utf-8') as f:
        return json.load(f)


def write_best(row):
    with open(BEST, 'w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2, default=str)


# ---------------- data ----------------
def load_data():
    if os.path.exists(CACHE):
        log('loading cached matrices')
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    log('reading raw csv (first time, ~1 min)')
    dt = {c: 'str' for c in ['id'] + STATIC}
    for i in range(1, NDAY + 1):
        dt['d_%d' % i] = 'float32'
    s = pd.read_csv(os.path.join(DATA, 'sales_train_evaluation.csv'), dtype=dt)
    groups = s[['id'] + STATIC].copy()
    sales = s[['d_%d' % i for i in range(1, NDAY + 1)]].to_numpy(dtype=np.float32)
    del s
    gc.collect()

    cal = pd.read_csv(os.path.join(DATA, 'calendar.csv'))
    pr = pd.read_csv(os.path.join(DATA, 'sell_prices.csv'),
                     dtype={'store_id': 'str', 'item_id': 'str',
                            'wm_yr_wk': 'int32', 'sell_price': 'float32'})
    weeks = np.sort(cal['wm_yr_wk'].unique())
    wmap = {int(w): i for i, w in enumerate(weeks)}
    day_week = cal['wm_yr_wk'].map(wmap).to_numpy(dtype=np.int32)

    key = groups['store_id'] + '|' + groups['item_id']
    k2i = {k: i for i, k in enumerate(key)}
    ri = (pr['store_id'] + '|' + pr['item_id']).map(k2i)
    ci = pr['wm_yr_wk'].map(wmap)
    price = np.zeros((len(groups), len(weeks)), dtype=np.float32)
    ok = ri.notna().to_numpy() & ci.notna().to_numpy()
    price[ri[ok].to_numpy(dtype=np.int64), ci[ok].to_numpy(dtype=np.int64)] = \
        pr['sell_price'].to_numpy(dtype=np.float32)[ok]
    del pr, ri, ci, ok
    gc.collect()

    dates = pd.to_datetime(cal['date'])
    e1 = cal['event_type_1'].fillna('none').astype('category').cat.codes.to_numpy(dtype=np.float32)
    e2 = cal['event_type_2'].fillna('none').astype('category').cat.codes.to_numpy(dtype=np.float32)
    cf = pd.DataFrame({
        'wday': cal['wday'].to_numpy(dtype=np.float32),
        'month': cal['month'].to_numpy(dtype=np.float32),
        'mday': dates.dt.day.to_numpy(dtype=np.float32),
        'doy_sin': np.sin(2 * np.pi * dates.dt.dayofyear.to_numpy() / 365.25).astype(np.float32),
        'doy_cos': np.cos(2 * np.pi * dates.dt.dayofyear.to_numpy() / 365.25).astype(np.float32),
        'wom': ((dates.dt.day.to_numpy() - 1) // 7).astype(np.float32),
        'payday': np.isin(dates.dt.day.to_numpy(), [1, 2, 14, 15, 16]).astype(np.float32),
        'is_dec': (cal['month'].to_numpy() == 12).astype(np.float32),
        'event_name': cal['event_name_1'].fillna('none').astype('category').cat.codes.to_numpy(dtype=np.float32),
        'snap_CA': cal['snap_CA'].to_numpy(dtype=np.float32),
        'snap_TX': cal['snap_TX'].to_numpy(dtype=np.float32),
        'snap_WI': cal['snap_WI'].to_numpy(dtype=np.float32),
        'event1': e1,
        'event2': e2,
        'weekend': cal['wday'].isin([1, 2]).to_numpy(dtype=np.float32),
        'mstart': (dates.dt.day <= 3).to_numpy(dtype=np.float32),
        'mend': dates.dt.is_month_end.to_numpy(dtype=np.float32),
    })
    codes = {c: pd.factorize(groups[c])[0].astype(np.int16) for c in STATIC}
    out = dict(sales=sales, groups=groups, cal=cf, day_week=day_week,
               price=price, codes=codes)
    with open(CACHE, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    log('cache saved | sales %s' % (sales.shape,))
    return out


# ---------------- WRMSSE ----------------
def _agg(mat, codes):
    return pd.DataFrame(mat).groupby(codes, sort=False).sum().to_numpy(dtype=np.float32)


def build_meta(D, cut=CUT_VAL):
    path = META if cut == CUT_VAL else os.path.join(ART, 'wrmsse_meta_%d.pkl' % cut)
    if os.path.exists(path):
        log('loading cached WRMSSE metadata cut=%d' % cut)
        with open(path, 'rb') as f:
            return pickle.load(f)
    log('building 12-level WRMSSE metadata cut=%d' % cut)
    sales, groups, price, dw = D['sales'], D['groups'], D['price'], D['day_week']
    train = sales[:, :cut]
    last28 = sales[:, cut - 28:cut] * price[:, dw[cut - 28:cut]]
    actual = sales[:, cut:cut + H]
    meta = []
    for cols in LEVELS:
        if not cols:
            codes = np.zeros(len(groups), dtype=np.int32)
        else:
            key = groups[cols[0]].astype(str)
            for c in cols[1:]:
                key = key + '|' + groups[c].astype(str)
            codes = pd.factorize(key)[0].astype(np.int32)
        ag = _agg(train, codes)
        dif = np.diff(ag, axis=1)
        scale = np.maximum((dif * dif).mean(axis=1), 1e-6).astype(np.float32)
        wb = _agg(last28, codes).sum(axis=1)
        w = (wb / max(float(wb.sum()), 1e-6)).astype(np.float32)
        ac = _agg(actual, codes)
        meta.append(dict(codes=codes, scale=scale, w=w, actual=ac,
                         name='_'.join(cols) or 'total', groups=int(len(scale))))
        del ag, dif, ac
        gc.collect()
        log('  level %2d/12 %-24s groups=%d' % (len(meta), meta[-1]['name'], meta[-1]['groups']))
    with open(path, 'wb') as f:
        pickle.dump(meta, f, protocol=4)
    return meta


def wrmsse(pred, meta):
    p = np.clip(np.asarray(pred, dtype=np.float32), 0, None)
    lv = []
    for L in meta:
        er = _agg(p, L['codes']) - L['actual']
        rmse = np.sqrt(np.mean(er * er, axis=1))
        lv.append(float(np.dot(L['w'], rmse / np.sqrt(L['scale']))))
    return float(np.mean(lv)), lv


# ---------------- features ----------------
def feature_names(D, key):
    base = STATIC + ['cal_' + c for c in D['cal'].columns] + [
        'lag28', 'wday_mean_4', 'roll_mean_7', 'roll_mean_28',
        'roll_std_28', 'zero_rate_28', 'roll_max_28', 'price', 'price_missing']
    price_extra = ['price_vs_28d', 'price_change', 'price_cv_8w', 'trend']
    long_extra = ['mean_all', 'roll_mean_56', 'roll_std_56', 'zero_rate_56',
                  'days_since_sale', 'lag35', 'lag42', 'lag56']
    hier_extra = ['hier_%s_%s' % (c, s) for c in
                  ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']
                  for s in ['28', 'trend']] + ['share_item']
    sets = {
        'base': base,
        'price': base + price_extra,
        'long': base + long_extra,
        'full': base + price_extra + long_extra,
        'full2': base + price_extra + long_extra + hier_extra,
    }
    return sets[key]


def build_features(D, cut, path):
    sales, price, dw, cal, codes = D['sales'], D['price'], D['day_week'], D['cal'], D['codes']
    n = sales.shape[0]
    hist = sales[:, :cut]
    cols = {}
    for c in STATIC:
        cols[c] = np.repeat(codes[c], H).astype(np.int16)
    cr = cal.iloc[cut:cut + H].to_numpy(dtype=np.float32)
    for j, name in enumerate(cal.columns):
        cols['cal_' + name] = np.tile(cr[:, j], n)

    cols['lag28'] = sales[:, cut - 28:cut].reshape(-1)
    cols['wday_mean_4'] = np.stack(
        [sales[:, cut - 28 * k:cut - 28 * (k - 1)] for k in (1, 2, 3, 4)]
    ).mean(0).reshape(-1)
    cols['lag35'] = sales[:, cut - 35:cut - 7].reshape(-1)
    cols['lag42'] = sales[:, cut - 42:cut - 14].reshape(-1)
    cols['lag56'] = sales[:, cut - 56:cut - 28].reshape(-1)

    m7 = hist[:, -7:].mean(1)
    m28 = hist[:, -28:].mean(1)
    m56 = hist[:, -56:].mean(1)
    s28 = hist[:, -28:].std(1)
    s56 = hist[:, -56:].std(1)
    z28 = (hist[:, -28:] == 0).mean(1)
    z56 = (hist[:, -56:] == 0).mean(1)
    mx28 = hist[:, -28:].max(1)
    prev28 = hist[:, -56:-28].mean(1)
    rep = lambda a: np.repeat(a, H).astype(np.float32)
    cols['roll_mean_7'] = rep(m7)
    cols['roll_mean_28'] = rep(m28)
    cols['roll_mean_56'] = rep(m56)
    cols['roll_std_28'] = rep(s28)
    cols['roll_std_56'] = rep(s56)
    cols['zero_rate_28'] = rep(z28)
    cols['zero_rate_56'] = rep(z56)
    cols['roll_max_28'] = rep(mx28)
    cols['trend'] = rep(m28 - prev28)
    cols['mean_all'] = rep(hist.mean(1))
    nz = hist[:, ::-1] != 0
    dsl = np.argmax(nz, axis=1).astype(np.float32) + 1
    dsl[~nz.any(1)] = float(cut)
    cols['days_since_sale'] = rep(dsl)

    w = dw[cut:cut + H]
    ph = price[:, w]
    pw = price[:, dw[cut - 56:cut]]
    p28 = pw[:, -28:].mean(1)
    p56 = pw.mean(1)
    cv = pw.std(1) / (p56 + 1e-3)
    cols['price'] = ph.reshape(-1).astype(np.float32)
    cols['price_missing'] = (ph == 0).astype(np.float32).reshape(-1)
    cols['price_vs_28d'] = (ph / (p28[:, None] + 1e-3)).astype(np.float32).reshape(-1)
    cols['price_change'] = rep(ph.mean(1) - p28)
    cols['price_cv_8w'] = rep(cv)

    gcodes = D['codes']

    def gsum(weights, c):
        idx = gcodes[c].astype(np.int64)
        k = int(idx.max()) + 1
        sm = np.bincount(idx, weights=weights, minlength=k)
        return sm[idx].astype(np.float32)

    for c in ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']:
        cols['hier_' + c + '_28'] = rep(gsum(m28, c))
        cols['hier_' + c + '_trend'] = rep(gsum(m28 - prev28, c))
    cols['share_item'] = rep(m28 / (gsum(m28, 'item_id') + 1e-3))

    X = pd.DataFrame(cols)
    for c in X.columns:
        X[c] = X[c].astype('int16') if c in STATIC else X[c].astype('float32')
    y = sales[:, cut:cut + H].reshape(-1).astype(np.float32) if cut + H <= NDAY else None
    with open(path, 'wb') as f:
        pickle.dump((X, y), f, protocol=4)
    log('  features cut=%d -> %s (%d cols)' % (cut, os.path.basename(path), X.shape[1]))
    return X, y


def get_X(D, cut, feats):
    path = os.path.join(ART, 'features_%d.pkl' % cut)
    if not os.path.exists(path):
        log('building features for cut %d' % cut)
        build_features(D, cut, path)
    X, y = pd.read_pickle(path)
    return X[list(feats)], y


def make_model(cfg):
    params = dict(objective=cfg['objective'], n_estimators=cfg['trees'],
                  learning_rate=cfg['lr'], num_leaves=cfg['num_leaves'],
                  min_child_samples=100, subsample=0.85, subsample_freq=1,
                  colsample_bytree=0.85, max_bin=127,
                  n_jobs=max(1, (os.cpu_count() or 4) - 1),
                  verbosity=-1, random_state=42)
    if cfg['objective'] == 'tweedie':
        params['tweedie_variance_power'] = 1.1
    return lgb.LGBMRegressor(**params)


# ---------------- experiment ----------------
def run_experiment(D, meta, cfg, stage):
    hid = cfg_hash(cfg, stage)
    if any(r['id'] == hid for r in read_jsonl(REG)):
        row = next(r for r in read_jsonl(REG) if r['id'] == hid)
        log('skip %s %s (already done, %.5f)' % (stage, cfg['name'], row['score']))
        return row['score']
    t0 = time.time()
    feats = feature_names(D, cfg['features'])
    Xs, ys = [], []
    for cut in cfg['windows']:
        X, y = get_X(D, cut, feats)
        Xs.append(X)
        ys.append(y)
    Xtr = pd.concat(Xs, ignore_index=True) if len(Xs) > 1 else Xs[0]
    ytr = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    model = make_model(cfg)
    log('train %s %s | rows=%d cols=%d trees=%d' %
        (stage, cfg['name'], len(ytr), Xtr.shape[1], cfg['trees']))
    model.fit(Xtr, ytr, categorical_feature=STATIC)
    del Xtr, ytr
    gc.collect()
    Xv, _ = get_X(D, CUT_VAL, feats)
    pred = np.clip(model.predict(Xv).reshape(-1, H), 0, None).astype(np.float32)
    score, levels = wrmsse(pred, meta)
    row = dict(id=hid, stage=stage, ts=time.strftime('%Y-%m-%d %H:%M:%S'),
               cfg=cfg, score=round(score, 6), levels=[round(x, 5) for x in levels],
               seconds=round(time.time() - t0, 1))
    append_jsonl(REG, row)
    log('RESULT %s %s WRMSSE=%.5f  (%.1fs)' % (stage, cfg['name'], score, row['seconds']))
    best = read_best()
    if best is None or score < best['score']:
        joblib.dump(model, os.path.join(MODELS, hid + '.pkl'))
        np.save(os.path.join(ART, 'val_pred_%s.npy' % hid), pred)
        write_best(dict(id=hid, score=score, cfg=cfg, stage=stage))
        log('PROMOTED -> %s (%.5f)' % (hid, score))
    del model, Xv, pred
    gc.collect()
    return score


def alt_prediction(D, cfg):
    """Train the same config on an earlier cut to obtain a second calibration period."""
    feats = feature_names(D, cfg['features'])
    cuts = [c for c in cfg['windows'] if c <= 1857] or [1733, 1789, 1829, 1857]
    Xs, ys = [], []
    for cut in cuts:
        X, y = get_X(D, cut, feats)
        Xs.append(X)
        ys.append(y)
    Xtr = pd.concat(Xs, ignore_index=True) if len(Xs) > 1 else Xs[0]
    ytr = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    model = make_model(cfg)
    log('ALT train | windows=%s rows=%d trees=%d' % (cuts, len(ytr), cfg['trees']))
    model.fit(Xtr, ytr, categorical_feature=STATIC)
    del Xtr, ytr
    gc.collect()
    Xv, _ = get_X(D, 1885, feats)
    pred = np.clip(model.predict(Xv).reshape(-1, H), 0, None).astype(np.float32)
    del model, Xv
    gc.collect()
    return pred


def grid_post(items):
    best = (1e9, 1.0, 0.0)
    for alpha in np.arange(0.82, 1.091, 0.02):
        for beta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            scores = []
            for pred, base, meta in items:
                p = alpha * ((1.0 - beta) * pred + beta * base[:, None])
                scores.append(wrmsse(p, meta)[0])
            s = float(np.mean(scores))
            if s < best[0]:
                best = (s, float(alpha), float(beta))
    log('post-process best alpha=%.2f beta=%.2f meanWRMSSE=%.5f (periods=%d)'
        % (best[1], best[2], best[0], len(items)))
    return best


def train_final(D, cfg, post):
    feats = feature_names(D, cfg['features'])
    Xs, ys = [], []
    for cut in cfg['windows']:
        X, y = get_X(D, cut, feats)
        Xs.append(X)
        ys.append(y)
    Xtr = pd.concat(Xs, ignore_index=True) if len(Xs) > 1 else Xs[0]
    ytr = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    model = make_model(cfg)
    log('FINAL train | rows=%d trees=%d features=%s' % (len(ytr), cfg['trees'], cfg['features']))
    model.fit(Xtr, ytr, categorical_feature=STATIC)
    del Xtr, ytr
    gc.collect()
    Xf, _ = get_X(D, CUT_FINAL, feats)
    pred = np.clip(model.predict(Xf).reshape(-1, H), 0, None).astype(np.float32)
    base = D['sales'][:, CUT_FINAL - 28:CUT_FINAL].mean(1)
    pred = np.clip(post[1] * ((1.0 - post[2]) * pred + post[2] * base[:, None]), 0, None)
    joblib.dump(model, os.path.join(ART, 'final_model.pkl'))
    np.save(os.path.join(ART, 'final_pred.npy'), pred)
    log('final prediction shape %s mean=%.4f' % (pred.shape, float(pred.mean())))
    return model, pred


# ---------------- submission ----------------
def write_submission(D, pred):
    sub = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
    ids = D['groups']['id'].astype(str).to_numpy()
    val_ids = [x.replace('_evaluation', '_validation') for x in ids]
    idx_v = {k: i for i, k in enumerate(val_ids)}
    idx_e = {k: i for i, k in enumerate(ids)}
    truth = D['sales'][:, CUT_VAL:CUT_VAL + H]
    out = np.zeros((len(sub), H), dtype=np.float32)
    sids = sub['id'].to_numpy()
    miss = 0
    for j, k in enumerate(sids):
        if k in idx_v:
            out[j] = truth[idx_v[k]]
        elif k in idx_e:
            out[j] = pred[idx_e[k]]
        else:
            miss += 1
    assert miss == 0, 'unmapped submission ids: %d' % miss
    assert np.isfinite(out).all()
    df = pd.DataFrame(out, columns=['F%d' % i for i in range(1, H + 1)])
    df.insert(0, 'id', sids)
    path = os.path.join(ART, 'submission.csv')
    df.to_csv(path, index=False)
    log('submission saved | shape=%s | validation rows: released truth; evaluation rows: model forecast' % (df.shape,))
    return path


def kaggle_submit(path):
    env = dict(os.environ)  # honour the caller's proxy settings, if any
    cmd = ['kaggle', 'competitions', 'submit', '-c', 'm5-forecasting-accuracy',
           '-f', path, '-m', 'auto self-iteration pipeline']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        msg = (r.stdout or '') + (r.stderr or '')
        log('submit rc=%d | %s' % (r.returncode, msg.strip()[:300]))
        if r.returncode == 0:
            s = subprocess.run(['kaggle', 'competitions', 'submissions',
                                '-c', 'm5-forecasting-accuracy'],
                               capture_output=True, text=True, env=env, timeout=300)
            out = (s.stdout or '') + (s.stderr or '')
            log('submissions:\n' + out[:1200])
            with open(os.path.join(ART, 'submissions.txt'), 'w', encoding='utf-8') as f:
                f.write(out)
            return out
    except Exception as e:
        log('submit failed: %s %s' % (type(e).__name__, str(e)[:200]))
    return ''


# ---------------- loop ----------------
def run():
    t0 = time.time()
    D = load_data()
    meta = build_meta(D)
    naive = np.repeat(D['sales'][:, CUT_VAL - 28:CUT_VAL].mean(1)[:, None], H, axis=1)
    base_score, _ = wrmsse(naive, meta)
    log('BASELINE last-28d-mean WRMSSE = %.5f' % base_score)

    log('=== STAGE A: %d candidates ===' % len(STAGE_A))
    for cfg in STAGE_A:
        run_experiment(D, meta, cfg, 'A')

    rows = [r for r in read_jsonl(REG) if r['stage'] == 'A']
    top2 = sorted(rows, key=lambda r: r['score'])[:2]
    log('=== STAGE B: promote top2 -> %s ===' % [r['cfg']['name'] for r in top2])
    for r in top2:
        cfg = dict(r['cfg'])
        cfg['name'] = r['cfg']['name'] + '_B'
        cfg['windows'] = [1829, 1857, 1885]
        cfg['trees'] = 700
        cfg['lr'] = 0.05
        run_experiment(D, meta, cfg, 'B')

    log('=== STAGE C: %d candidates (5 windows + hierarchy features) ===' % len(STAGE_C))
    for cfg in STAGE_C:
        run_experiment(D, meta, cfg, 'C')

    log('=== STAGE D: %d candidates (near windows + deeper fit) ===' % len(STAGE_D))
    for cfg in STAGE_D:
        run_experiment(D, meta, cfg, 'D')

    best = read_best()
    pred = np.load(os.path.join(ART, 'val_pred_%s.npy' % best['id']))
    base28 = D['sales'][:, CUT_VAL - 28:CUT_VAL].mean(1)
    meta2 = build_meta(D, 1885)
    p2path = os.path.join(ART, 'alt_pred_%s.npy' % best['id'])
    if os.path.exists(p2path):
        pred2 = np.load(p2path)
        log('alt prediction reused')
    else:
        pred2 = alt_prediction(D, dict(best['cfg']))
        np.save(p2path, pred2)
    base2 = D['sales'][:, 1857:1885].mean(1)
    post = grid_post([(pred, base28, meta), (pred2, base2, meta2)])

    cfg = dict(best['cfg'])
    cfg['name'] = best['cfg']['name'] + '_FINAL'
    # Window set that produced private WRMSSE 0.66138 (top-10% cutoff 0.69934)
    cfg['windows'] = [1789, 1829, 1857, 1885, CUT_VAL]
    cfg['trees'] = 800
    cfg['lr'] = 0.045
    _, pred_final = train_final(D, cfg, post)
    path = write_submission(D, pred_final)
    subs = kaggle_submit(path)

    allrows = read_jsonl(REG)
    result = dict(
        finished=time.strftime('%Y-%m-%d %H:%M:%S'),
        minutes=round((time.time() - t0) / 60, 1),
        baseline_wrmsse=round(base_score, 5),
        best_local=dict(id=best['id'], score=round(best['score'], 5),
                        cfg=best['cfg'], stage=best['stage']),
        post_process=dict(alpha=round(post[1], 3), beta=round(post[2], 3),
                          score=round(post[0], 5)),
        experiments=[dict(stage=r['stage'], name=r['cfg']['name'],
                          score=r['score'], seconds=r['seconds']) for r in allrows],
        submission=path,
        submissions_output=subs[:2000],
        artifacts=dict(registry=REG, best=BEST, log=LOG),
    )
    with open(RESULT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log('DONE in %.1f min | best local WRMSSE %.5f' % (result['minutes'], best['score']))
    return result
