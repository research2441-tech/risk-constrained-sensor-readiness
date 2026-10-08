"""Prepare real environmental and irradiance traces used by the trace-driven emulator.

Sources (all public):
  * Beijing Multi-Site Air-Quality data (12 stations, hourly, 2013-03-01..2017-02-28):
    CO, NO2, temperature, dew point, wind speed, rain.
  * NSRDB PSM v3 (2017) and PSM v4 (2023) half-hourly GHI and clear-sky GHI at 40.53 N, 108.54 W
    (files distributed with the pvlib-python test suite).
  * BSRN Payerne (June 2016) one-minute GHI, used as an analogue library for sub-30-min variability.

Outputs: data/beijing.npz, data/kc30.npz, data/bsrn_analogues.npz
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import os, glob, sys
import numpy as np, pandas as pd, pvlib

RAW = sys.argv[1] if len(sys.argv) > 1 else _HERE
OUT = sys.argv[2] if len(sys.argv) > 2 else _HERE
os.makedirs(OUT, exist_ok=True)

STATIONS = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng',
            'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']


def beijing():
    frames, miss = [], {}
    for s in STATIONS:
        f = glob.glob(os.path.join(RAW, 'submission_air_quality/data', f'PRSA_Data_{s}_*.csv'))[0]
        d = pd.read_csv(f)
        d.index = pd.to_datetime(d[['year', 'month', 'day', 'hour']])
        d = d[['CO', 'NO2', 'TEMP', 'DEWP', 'WSPM', 'RAIN']].astype(float)
        miss[s] = d.isna().mean().to_dict()
        d = d.interpolate(limit=6, limit_direction='both').ffill().bfill()
        frames.append(d)
    idx = frames[0].index
    arr = np.stack([f.reindex(idx).values for f in frames], axis=1)  # (T, 12, 6)
    np.savez_compressed(os.path.join(OUT, 'beijing.npz'), data=arr.astype(np.float32),
                        time=idx.values.astype('datetime64[h]').astype(np.int64),
                        vars=np.array(['CO', 'NO2', 'TEMP', 'DEWP', 'WSPM', 'RAIN']),
                        stations=np.array(STATIONS))
    pd.DataFrame(miss).T.to_csv(os.path.join(OUT, 'beijing_missing.csv'))
    print('beijing', arr.shape)


def kc30():
    d = os.path.join(RAW, 'pvrepo/tests/data')
    out = {}
    # PSM3 2017
    p3 = pd.read_csv(os.path.join(d, 'test_read_psm3.csv'), skiprows=2)
    ghi, cs = p3['GHI'].values.astype(float), p3['Clearsky GHI'].values.astype(float)
    t3 = p3['Temperature'].values.astype(float)
    out['y2017'] = (ghi, cs)
    p4, meta = pvlib.iotools.read_nsrdb_psm4(os.path.join(d, 'test_read_psm4.csv'))
    out['y2023'] = (p4['ghi'].values.astype(float), p4['ghi_clear'].values.astype(float))
    res = {}
    for k, (g, c) in out.items():
        kc = np.where(c > 20, g / np.maximum(c, 1), np.nan)
        kc = np.clip(kc, 0, 1.4)
        res[k] = kc[:17520].astype(np.float32)
        print(k, 'kc mean (daylight)', np.nanmean(kc).round(3), len(kc))
    np.savez_compressed(os.path.join(OUT, 'kc30.npz'), **res)


def bsrn():
    d = os.path.join(RAW, 'pvrepo/tests/data')
    df, meta = pvlib.iotools.read_bsrn(os.path.join(d, 'bsrn-pay0616.dat.gz'))
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], altitude=meta['altitude'])
    cs = loc.get_clearsky(df.index, model='ineichen')['ghi']
    kc = (df['ghi'] / cs.where(cs > 50)).clip(0, 1.4)
    kc = kc.values
    blocks = kc[: len(kc) // 30 * 30].reshape(-1, 30)
    ok = ~np.isnan(blocks).any(1)
    blocks = blocks[ok]
    means = blocks.mean(1)
    devs = blocks - means[:, None]
    np.savez_compressed(os.path.join(OUT, 'bsrn_analogues.npz'), means=means.astype(np.float32),
                        devs=devs.astype(np.float32))
    print('bsrn analogue blocks', len(means), 'mean kc', means.mean().round(3))


if __name__ == '__main__':
    beijing(); kc30(); bsrn()
