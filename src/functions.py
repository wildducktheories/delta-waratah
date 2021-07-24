import numpy as np
import os
import pandas as pd
import re
import statsmodels.api as sm
import sys

from datetime import datetime
from statsmodels.regression.rolling import RollingOLS

def load_data(fn):
    df=pd.read_csv(fn).rename({
        'Date': 'date',
        'Overseas-acquired': 'overseas',
        'Overseas acquired': 'overseas',
        'Known local source': 'local',
        'Unknown local source (community spread)': 'unknown',
        'Interstate travel': 'interstate',
        'Known local source inside HQ': 'local-hq',
        'Under investigation': 'under-investigation'
    }, axis=1)
    try:
        df["local-hq"] is None
    except KeyError as e:
        df["local-hq"] = 0
    df=calc_cumulative(df)
    return edit_dates(df)

def load_quarantine(fn):
    df=pd.read_csv(fn).rename({
        'Hotel Quarantine': 'hotel-quarantine',
        'Local Quarantine / Isolation': 'local-quarantine',
        'No Quarantine / Unspecified': 'no-quarantine',
        'Not Announced / Under investigation': 'under-investigation'
    }, axis=1)
    df['total']=df['under-investigation']+df['no-quarantine']
    df['cumulative'] = df['total'].cumsum()
    return edit_dates(df)

def rewrite_date(d):
    p = d.split('/')
    return '-'.join(['20'+p[2], p[1], p[0]])

def calc_cumulative(df):
    df['total']=df['local']+df['unknown']+df['under-investigation']+df['local-hq']
    df['cumulative'] = df['total'].cumsum()
    return df

def select_outbreak(df, generation_days=5):
    subset_df = df.copy().reset_index(drop=True)
    start_total, start_cumulative = subset_df[['total', 'cumulative']].head(1).values[0]
    subset_df['cumulative'] = subset_df['cumulative'] - start_cumulative + start_total

    endog=np.log(subset_df['cumulative'])
    exog=sm.add_constant(subset_df.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    subset_df['ols-growth-rate']=((np.exp(model.fit().params['x1'])-1)*100)
    subset_df['ratio-growth-rate']=((subset_df['cumulative']/subset_df['cumulative'].shift(generation_days))**(1/generation_days)-1)*100
    subset_df['Reff'] = ((subset_df['ols-growth-rate']+100)/100)**generation_days
    subset_df['doubling-period'] = np.log(2)/np.log(((subset_df['ols-growth-rate']+100)/100))

    subset_df['ols-growth-rate-median'] = subset_df['ols-growth-rate'].rolling(generation_days, min_periods=1).median()
    subset_df['ols-growth-rate-min'] = subset_df['ols-growth-rate'].rolling(generation_days, min_periods=1).min()
    subset_df['ols-growth-rate-max'] = subset_df['ols-growth-rate'].rolling(generation_days, min_periods=1).max()

    endog=np.log((100+subset_df['ols-growth-rate'])/100)
    exog=sm.add_constant(subset_df.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    subset_df['ols-growth-rate-decay']=((np.exp(model.fit().params['x1'])-1)*100)

    shift_df=subset_df.shift(1)
    subset_df['one-day-error']=subset_df['cumulative']-np.round(shift_df['cumulative']*((shift_df['ols-growth-rate']+100)/100))
    subset_df['one-day-projection-cumulative']=np.round(subset_df['cumulative']*((subset_df['ols-growth-rate']+100)/100))
    subset_df['one-day-projection-total']=subset_df['one-day-projection-cumulative']-subset_df['cumulative']


    subset_df['median']=(((subset_df['ols-growth-rate-median']+100)/100)**7*subset_df['cumulative']).shift(7)
    subset_df['min']=(((subset_df['ols-growth-rate-min']+100)/100)**7*subset_df['cumulative']).shift(7)
    subset_df['max']=(((subset_df['ols-growth-rate-max']+100)/100)**7*subset_df['cumulative']).shift(7)
    subset_df['last']=(((subset_df['ols-growth-rate']+100)/100)**7*subset_df['cumulative']).shift(7)

    subset_df["err"] = modeling_errors(subset_df, None)
    return subset_df

def edit_dates(df):
    newyear=df['date'][df['date']=='01/01'].index.values
    if len(newyear) > 0:
        df.loc[df.index < newyear[0],['date']]=df['date'].loc[df.index < newyear[0]]+'/20'
    else:
        newyear=[0]
    df.loc[df.index >= newyear[0],['date']]=df['date'].loc[df.index >= newyear[0]]+'/21'
    df['date']=df['date'].apply(rewrite_date)
    return df

def load_vic_data(fn):
    df=pd.read_csv(fn).rename({
        'Acquired in Australia, unknown source': 'unknown',
        'Contact with a confirmed case	': 'local',
        'Travel overseas': 'overseas'
    })

    df['count']=1
    df=df.pivot_table(index=['diagnosis_date'], columns=['acquired'], values=['count'], aggfunc=np.sum)

    tmp=df
    df=pd.DataFrame(data=df.values, columns=['unknown', 'local', 'overseas'])
    df[df.loc[:,['unknown']].isna()]=0
    df[df.loc[:,['local']].isna()]=0
    df[df.loc[:,['overseas']].isna()]=0
    df['under-investigation']=0

    df=pd.DataFrame(data=df.values, columns=['unknown', 'local', 'overseas', 'under-investigation'], dtype=int)
    df['date']=tmp.index

    df['total']=df['local']+df['unknown']+df['under-investigation']
    df['cumulative'] = df['total'].cumsum()
    return df

def amnesic_growth(rate=1.5, days=100, window_size=14):
    window = np.zeros(shape=(window_size))
    population = 0.0
    cumulative = 0
    for d in range(0,days):
        old_population = window[d%window_size]
        if (population-window[d%window_size]) > 0:
            next_population = (population-window[d%window_size]) * (rate-1) + population
        elif d == 0:
            next_population = 1
        else:
            next_population = population
        total = int(next_population) - int(population)
        cumulative += total
        prev_population = population
        population = next_population
        window[d%window_size] = next_population
        yield d, total, cumulative, population, prev_population, old_population

def modeling_errors(df, clip=None):
    err = (df['min']-df['cumulative'])/(df['cumulative']-df['cumulative'].shift(7))*100
    if clip:
        return err.apply(lambda v: clip if v > clip else v)
    else:
        return err

def derive_growth_params(df, generation_days=5):
    endog=np.log(df['ols-growth-rate'])
    exog=sm.add_constant(df.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    params=np.exp(model.fit().params.mean())
    return (params["const"], params["x1"])


def project_cumulative(df, days, rate):
    last = df.tail(1)
    max_index = last.index.values[0]
    max_cumulative = last.cumulative.values[0]
    df = df.reindex(range(0, max_index+days+1))

    growth = ((rate+100)/100)**(np.array(range(1, days+1)))
    df.loc[df.index > max_index, 'cumulative'] = np.int32(max_cumulative*growth)
    df.loc[df.index > max_index, 'total'] = df.loc[df.index > max_index, 'cumulative'] - df.loc[df.index > max_index-1, 'cumulative'].shift(1)
    return df

def project_ols_growth_rate_min(df, days, growth_decay_rate):
    last = df.tail(1)
    index = last.index.values[0]
    cumulative = last.cumulative.values[0]
    ols_growth_rate = last['ols-growth-rate-min'].values[0]

    tuples = []
    for d in range(1, days+1):
        ols_growth_rate = ols_growth_rate*growth_decay_rate
        cumulative = cumulative * (ols_growth_rate+100)/100
        tuples.append(np.array([cumulative]))

    df = df.reindex(range(0, index+days+1))
    df.loc[df.index > index, 'cumulative'] = tuples
    df.loc[df.index > index, 'total'] = df.loc[df.index > index, 'cumulative'] - df.loc[df.index > index-1, 'cumulative'].shift(1)

    return df

def horizon(df, days, steps=100):

    last = df.tail(1)
    index = last.index.values[0]
    cumulative = last["cumulative"].values[0]
    ols_growth_rate_min = last["ols-growth-rate-min"].values[0]
    ols_growth_rate_max = last["ols-growth-rate-max"].values[0]

    max_rate = np.sqrt(ols_growth_rate_max/ols_growth_rate_min) * ols_growth_rate_max
    min_rate = ols_growth_rate_min/5

    rate_step = (max_rate/min_rate)**(1/steps)

    c_rate = (((100+ols_growth_rate_min)/100)**days) * cumulative
    t_rate = np.round(c_rate*(1-100/(100+ols_growth_rate_min)))

    tuples=[]
    for s in range(0, steps):
        r = min_rate*(rate_step ** (s+1))
        c = (((100+r)/100)**days)*cumulative
        t = np.round(c*(1-100/(100+r)))
        e = (c_rate-c)/(c-cumulative)*100
        tuples.append(np.array([r,c,e,t]))

    df = pd.DataFrame(
        index=range(0,steps),
        columns=["ols-growth-rate", "cumulative", "err", "total"],
        data=tuples)
    return df[np.abs(df["err"])<=100]


def plot_derivatives(df, split=None, dataset="???"):

    out=pd.DataFrame(index=df.index)

    cumulative_max=int(np.round(df["cumulative"].max()))
    total_max=int(np.round(df["total"].max()))
    ols_growth_rate_max=int(np.round(df["ols-growth-rate"].max()))
    ols_growth_rate_decay_max=int(np.round(np.abs(df["ols-growth-rate-decay"]).max()))

    out["cumulative"]=df["cumulative"]/cumulative_max
    out["total"]=df["total"]/total_max
    out["growth"]=df["ols-growth-rate"]/ols_growth_rate_max
    out["decay"]=df["ols-growth-rate-decay"]/ols_growth_rate_decay_max

    if split:
        past   = out[out.index < split]
        future = out[out.index >= split]
    else:
        past = out
        future = None

    ax=past.plot(figsize=(10,10), title=f"Derivatives Past and Present ({dataset})")
    ax.grid()
    ax.legend([
        f"cumulative ({cumulative_max})",
        f"daily - first derivative ({total_max})",
        f"growth - 2nd derivive ({ols_growth_rate_max})",
        f"decay - 3rd derivtive ({ols_growth_rate_decay_max})"])
    ax.set_xlabel("day of outbreak")
    ax.set_ylabel("relative scale")

    if not future is None:
        ax.plot(future['cumulative'], linestyle='dotted')
        ax.plot(future['total'], linestyle='dotted')
        ax.plot(future['growth'], linestyle='dotted')
        ax.plot(future['decay'], linestyle='dotted')
    return ax


def sweep_downloads(date):

    DATE_ARCHIVE=f"archive/{date}"
    os.makedirs(DATE_ARCHIVE, exist_ok=True)
    DOWNLOADS=f"{os.environ['HOME']}/Downloads"

    pattern=re.compile("Last (14 days|6 months) \\((new|true)\\)( *\\([0-9]*\\))*")
    files = [f for f in os.listdir(DOWNLOADS) if pattern.match(f)]

    for f in files:
        src=f"{DOWNLOADS}/{f}"
        print(f"found \"{src}\"", file=sys.stderr)

    for f in files:
        src=f"{DOWNLOADS}/{f}"
        if f == 'Last 14 days (new).csv':
            nf = 'last-14-days-nsw.csv'
        elif f == 'Last 6 months (true).csv':
            nf = 'last-6-months-nsw.csv'
        else:
            print("rm {f}", file=sys.stderr)
            os.unlink(f"{src}")
            continue

        dst=f"{DATE_ARCHIVE}/{nf}"

        os.rename(src,dst)
        print(f"mv {src} {dst}", file=sys.stderr)

def update_df(base_df, update_df):
    update_df=update_df.copy()
    join_index=base_df.loc[base_df["date"] == update_df.head(1)['date'].values[0]].index.values[0]
    update_df['renumber'] = range(join_index, join_index+len(update_df))
    update_df=update_df.set_index('renumber', drop=True)
    base_df.update(update_df, overwrite=True)
    return calc_cumulative(base_df)

def exponent(rate):
    return np.log(factor(rate))

def factor(rate):
    return (rate+100)/100
