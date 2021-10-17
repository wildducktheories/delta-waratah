import numpy as np
import os
import pandas as pd
import re
import statsmodels.api as sm
import sys

from .PhasePlot import PhasePlot

from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from collections import OrderedDict

def append_recent_date(df, date, local):
    return df.append(
        {
            "date": date,
            "total": local,
            "cumulative": df.tail(1)["cumulative"].values[0]+local,
            "cumulative_corrected": df.tail(1)["cumulative_corrected"].values[0]+local,
            "correction": df.tail(1)["correction"].values[0]
        }, ignore_index=True)

def calc_cumulative(df):
    df['total']=df['local']+df['unknown']+df['under-investigation']+df['local-hq']
    df['cumulative'] = df['total'].cumsum()
    return df

def calculate_ltlc_params(df2, window=14):
    df=df2[df2.total != 0]
    endog=np.log(df['total'])
    exog=sm.add_constant(np.log(df['cumulative']))
    model = RollingOLS(endog=endog, exog=exog, window=window, min_nobs=window)
    params=model.fit().params
    df2.loc[params.index, "ltlc-intercept"] = params["const"]
    df2.loc[params.index, "ltlc-gradient"] = params["cumulative"]

def select_outbreak(df_in, generation_days=5):
    df = df_in.copy().reset_index(drop=True)
    start_total, start_cumulative = df[['total', 'cumulative']].head(1).values[0]
    df['cumulative'] = df['cumulative'] - start_cumulative + start_total

    endog=np.log(df['cumulative'])
    exog=sm.add_constant(df.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    df['ols-growth-rate']=((np.exp(model.fit().params['x1'])-1)*100)
    df['ratio-growth-rate']=((df['cumulative']/df['cumulative'].shift(generation_days))**(1/generation_days)-1)*100
    df['Reff-old'] = ((df['ols-growth-rate']+100)/100)**generation_days

    endog=np.log(df['total'].rolling(window=generation_days).mean())
    exog=sm.add_constant(df.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    df['ols-growth-rate-new']=((np.exp(model.fit().params['x1'])-1)*100)
    df['Reff'] = ((df['ols-growth-rate-new']+100)/100)**generation_days

    ols_growth_rate=df.loc[df['ols-growth-rate'].notna(), ['ols-growth-rate']].copy()

    df['doubling-period'] = np.log(2)/np.log(((ols_growth_rate+100)/100))

    df['ols-growth-rate-median'] = ols_growth_rate.rolling(generation_days, min_periods=1).median()
    df['ols-growth-rate-min'] = ols_growth_rate.rolling(generation_days, min_periods=1).min()
    df['ols-growth-rate-max'] = ols_growth_rate.rolling(generation_days, min_periods=1).max()
    df['linear-growth-rate'] = (df['total']/df['cumulative']*100)
    df['linear-growth-rate-max'] = df['linear-growth-rate'].rolling(window=5).max()
    df['linear-growth-rate-min'] = df['linear-growth-rate'].rolling(window=5).min()
    df['linear-growth-rate-mean'] = df['linear-growth-rate'].rolling(window=5).mean()
    df['linear-growth-rate-relative-error'] = ((df['linear-growth-rate']-df['ols-growth-rate'])/df['ols-growth-rate'])*100

    endog=np.log(ols_growth_rate)
    exog=sm.add_constant(ols_growth_rate.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    df['ols-growth-rate-decay']=rate(np.exp(model.fit().params['x1']))

    shift_df=df.shift(1)
    df['one-day-projection-cumulative']=np.round(df['cumulative']*((df['ols-growth-rate']+100)/100))
    df['one-day-projection-total']=df['one-day-projection-cumulative']-df['cumulative']
    df['one-day-error']=np.round(shift_df['cumulative']*((shift_df['ols-growth-rate']+100)/100)-df['cumulative'])
    df["one-day-relative-error"] = df["one-day-error"]/df["total"]*100


    df['median']=(((df['ols-growth-rate-median']+100)/100)**7*df['cumulative']).shift(7)
    df['min']=(((df['ols-growth-rate-min']+100)/100)**7*df['cumulative']).shift(7)
    df['max']=(((df['ols-growth-rate-max']+100)/100)**7*df['cumulative']).shift(7)
    df['last']=(((df['ols-growth-rate']+100)/100)**7*df['cumulative']).shift(7)

    df["7-day-delta"] = df["cumulative"]-df["cumulative"].shift(7)
    df["7-day-projection"] = np.round(df["min"])
    df["7-day-projection-relative-error"] = modeling_errors(df, None)
    df["7-day-projection-error"] = df["7-day-projection"]-df['cumulative']
    df["7-day-forward-projection-cumulative"] = np.int32(np.round(df["cumulative"]*(factor(df["ols-growth-rate-min"])**7)))
    df["7-day-forward-projection-total"] = np.int32(np.round(df["cumulative"]*(factor(df["ols-growth-rate-min"])**6)*(factor(df["ols-growth-rate-min"])-1)))

    calculate_ltlc_params(df)

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
    err = (df['7-day-projection']-df['cumulative'])/(df['7-day-delta'])*100
    if clip:
        return err.apply(lambda v: clip if v > clip else v)
    else:
        return err

def derive_growth_params(df, generation_days=5):
    endog=np.log(df['ols-growth-rate'])
    exog=sm.add_constant(df.index)
    model = RollingOLS(endog=endog, exog=exog, window=generation_days, min_nobs=generation_days)
    params=np.exp(model.fit().params.mean())
    return (params["const"], rate(params["x1"]))


def project_cumulative(df, days, rate):
    last = df.tail(1)
    max_index = last.index.values[0]
    max_cumulative = last.cumulative.values[0]
    df = df.reindex(range(0, max_index+days+1))

    growth = ((rate+100)/100)**(np.array(range(1, days+1)))
    df.loc[df.index > max_index, 'cumulative'] = np.int32(max_cumulative*growth)
    df.loc[df.index > max_index, 'total'] = df.loc[df.index > max_index, 'cumulative'] - df.loc[df.index > max_index-1, 'cumulative'].shift(1)
    return df

def project_ols_growth_rate_min(df, days, growth_decay_rate, growth_rate_column='ols-growth-rate-min'):
    last = df.tail(1)
    index = last.index.values[0]
    cumulative = last.cumulative.values[0]
    ols_growth_rate = last[growth_rate_column].values[0]
    date = last['date'].values[0]

    tuples = []
    dates = []
    for d in range(1, days+1):
        date = add_days(date,1)
        cumulative = cumulative * factor(ols_growth_rate)
        ols_growth_rate = ols_growth_rate * factor(growth_decay_rate)
        dates.append(date)
        tuples.append(cumulative)
    df = df.reindex(range(0, index+days+1))
    #df.loc[df.index > index, ['date']] = tuples[:,0]
    df.loc[df.index > index, 'cumulative'] = tuples
    df.loc[df.index > index, 'date'] = dates
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
        columns=["ols-growth-rate", "cumulative", "7-day-projection-relative-error", "total"],
        data=tuples)
    return df[np.abs(df["7-day-projection-relative-error"])<=100]


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

def rate(factor):
    return (factor-1)*100

def animate_phaseplot(df, offset, outbreak, fn):
    images=[]
    for i in range(offset, len(df)):
        view=df.head(i+1)
        date=view.tail(1).date.values[0]
        pp = PhasePlot(f"Daily Total (cases) vs 7-Day Projection Error (%) - {outbreak} - ({date})")
        idx=pp.add(view, offset=offset, legend=outbreak, color="C1") # 15
        pp.ax.set_xlim(-100,100)
        pp.ax.set_ylim(1,2000)
        pp.add_horizon(horizon(view.head(len(view)-7), 7), legend="Previous 7-Day horizon", color="red")
        pp.add_horizon(horizon(view, 7), legend="Current 7-Day horizon", color="blue")
        b=BytesIO()
        pp.ax.figure.savefig(b, format="png")
        images.append(Image.open(b))

    for i in range(0, 10):
        images.append(images[-1])

    images[0].save(fn, save_all=True, append_images=images[1:], loop=0, duration=300)


DATE_FORMAT="%Y-%m-%d"
def add_days(date, days):
    return (datetime.strptime(date, DATE_FORMAT)+timedelta(days=days)).strftime(DATE_FORMAT)

def animate_derivatives(df, outbreak, fn, decay_rate):
    images=[]
    for x in range(5, len(df)+1):
        df1=select_outbreak(project_ols_growth_rate_min(df.head(x), 150-x, decay_rate))
        date=df.head(x).tail(1)['date'].values[0]
        ax=plot_derivatives(df1, x, f"{outbreak} ({date})")
        b=BytesIO()
        ax.figure.savefig(b, format="png")
        images.append(Image.open(b))

    for i in range(0, 10):
        images.append(images[-1])

    images[0].save(fn, save_all=True, append_images=images[1:], loop=0, duration=300)

def plot_log_log_trend(ax, df, color, max_bounds, index):
    fit=pd.DataFrame()
    fit["cumulative"]=[r for r in range(1, max_bounds[0], int(max_bounds[0]/100))]
    coeffs=(df.tail(1)[['ltlc-intercept', 'ltlc-gradient']].values[0])
    fit["total"]=np.exp((coeffs[1]*np.log(fit["cumulative"]))+coeffs[0])
    ax.plot(fit["cumulative"], fit["total"], linestyle="dashed", color=color)
    legends = [t.get_text() for t in ax.get_legend().get_texts()]
    legends[index] = f"{legends[index]} - ltlc-gradient={round(coeffs[1],3)}"
    ax.legend(legends)
    slice=df[df.index > df.index.max()-14]
    ax.scatter(slice['cumulative'], slice['total'])

    return ax


def plot_log_log_day(day, vic, syd):
    max_bounds=(syd['cumulative'].max(),syd['total'].max()*7)
    slice = syd[syd.index <= day]
    future = syd[syd.index > day]
    vic_slice=vic[vic.index <= day]
    vic_future=vic[vic.index > day]

    ax=slice.plot(x="cumulative", y="total", figsize=(10,10))
    ax.plot(vic_slice["cumulative"], vic_slice["total"])

    ax.plot(future["cumulative"], future["total"], color="C0", linestyle="dotted")
    ax.plot(vic_future["cumulative"], vic_future["total"], color="C1", linestyle="dotted")

    _=ax.legend([ "Sydney 2021", "Melbourne 2020"])

    ax.set_title(f"log(total) vs log(cumulative) - day {day}")
    ax.set_ylabel("Daily New Cases (t)")
    ax.set_xlabel("Cumulative Cases (c)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(top=max_bounds[1])

    plot_log_log_trend(ax, syd[syd.index<=day], color="C0", max_bounds=max_bounds, index=0)
    plot_log_log_trend(ax, vic[vic.index<=day], color="C1", max_bounds=max_bounds, index=1)

    return ax

def animate_log_log_plot(vic, syd, fn):
    images=[]
    for i in range(15, vic.index.max()+1):
        ax=plot_log_log_day(i, vic, syd)
        b=BytesIO()
        ax.figure.savefig(b, format="png")
        images.append(Image.open(b))

    images[0].save(fn, save_all=True, append_images=images[1:], loop=0, duration=300)

def summary(df):
    def percentage_delta(df):
        shift=df.shift(1)
        return np.round((df - shift)/shift*100,2)


    def delta(df):
        shift=df.shift(1)
        return np.round((df - shift),3)

    arrow=lambda v : "↑" if v > 0 else "↓" if v < 0 else "-"
    f=lambda v: f"{arrow(v)}Δ {v}"
    g=lambda v: f'<span class="good">{f(v)}</span>' if v < 0 else f'<span class="bad">{f(v)}</span>'
    gg=lambda v: f'<span class="good">{v}</span>' if v < 0 else f'<span class="bad">{v}</span>'
    h=lambda v: f'<span class="bad">{f(v)}</span>' if v < 0 else f'<span class="good">{f(v)}</span>'
    hh=lambda v: f'<span class="bad">{v}</span>' if v < 0 else f'<span class="good">{v}</span>'

    columns=[
        "cumulative",
        "total",
        "correction",
        "one-day-error",
        "ols-growth-rate",
        "ols-growth-rate-decay",
        "doubling-period",
        "Reff-old",
        "Reff",
        "ltlc-gradient",
        "one-day-projection-total",
        "linear-growth-rate"
    ]

    stats=df[columns].tail(1)
    delta=delta(df[columns]).tail(1)
    percentage_delta(df[columns]).tail(1)
    delta_symbol=("Δ")

    slice=df[(df.date >= '2021-07-02')]
    gp0=derive_growth_params(slice[slice.index < slice.tail(1).index.values[0]])
    decay_rate0=gp0[1]
    gp1=derive_growth_params(slice)
    decay_rate1=gp1[1]

    summary="""
    <style>
        span.good {
            color: green;
        }
        span.bad {
            color: red;
        }
    </style>
    """+f"""
    <h1>Summary</h1>
    <br/>
    <pre>
    Date: {df.tail(1)['date'].values[0]} (#{df.tail(1).index.values[0]})

    New Cases Reported Today: {round(df.tail(1)["total"].values[0])} {g(delta["total"].values[0])}
    Cumulative (since 2021-06-16): {round(df.tail(1)["cumulative"].values[0])} {f(delta["cumulative"].values[0])}
    Corrections (since 2021-06-16): {round(df.tail(1)["correction"].values[0])} {f(delta["correction"].values[0])}

    Projection (from yesterday): {round(df.tail(2).head(1)["one-day-projection-total"].values[0])}
    Projection Error: {round(df.tail(1)["one-day-error"].values[0])} ({hh(round(df.tail(1)["one-day-relative-error"].values[0],1))}%)
    Projection (for tomorrow): {round(df.tail(1)["one-day-projection-total"].values[0])}
    Projection (for next week): {round(df.tail(1)["7-day-forward-projection-total"].values[0])}

    Cumulative Growth Rate: {round(df.tail(1)["ols-growth-rate"].values[0],2)}% per day {g(delta["ols-growth-rate"].values[0])}
    Linear Growth Rate : {round(df.tail(1)["linear-growth-rate"].values[0],2)}% per day {g(delta["linear-growth-rate"].values[0])}
    Reff: {round(df.tail(1)["Reff"].values[0],2)} {g(delta["Reff"].values[0])}
    Doubling Period:  {round(df.tail(1)["doubling-period"].values[0],1)} days {h(delta["doubling-period"].values[0])}

    Growth Decay Rate: {round(decay_rate1, 2)}% per day {g(round(decay_rate1 - decay_rate0, 2))}
    ln-ln Gradient: {round(df.tail(1)["ltlc-gradient"].values[0], 3)} {g(round(delta["ltlc-gradient"].values[0], 3))}
    </pre>"""+f"""<br/><br/>https://jonseymour.s3.amazonaws.com/delta-waratah/archive/{df.tail(1)['date'].values[0]}/sydney-outbreaks.html"""

    return summary

def weekly_summary(df):
    def percentage_delta(df):
        shift=df.shift(7)
        return np.round((df - shift)/shift*100,2)


    def delta(df):
        shift=df.shift(7)
        return np.round((df - shift),3)

    arrow=lambda v : "↑" if v > 0 else "↓" if v < 0 else "-"
    f=lambda v: f"{arrow(v)}Δ {v}"
    g=lambda v: f'<span class="good">{f(v)}</span>' if v < 0 else f'<span class="bad">{f(v)}</span>'
    gg=lambda v: f'<span class="good">{v}</span>' if v < 0 else f'<span class="bad">{v}</span>'
    h=lambda v: f'<span class="bad">{f(v)}</span>' if v < 0 else f'<span class="good">{f(v)}</span>'
    hh=lambda v: f'<span class="bad">{v}</span>' if v < 0 else f'<span class="good">{v}</span>'

    columns=[
        "cumulative",
        "total",
        "correction",
        "one-day-error",
        "ols-growth-rate",
        "ols-growth-rate-decay",
        "doubling-period",
        "Reff",
        "ltlc-gradient",
        "one-day-projection-total",
        "linear-growth-rate"
    ]

    stats=df[columns].tail(1)
    delta=delta(df[columns]).tail(1)
    percentage_delta(df[columns]).tail(1)
    delta_symbol=("Δ")

    slice=df[(df.date >= '2021-07-02')]
    gp0=derive_growth_params(slice[slice.index < slice.tail(8).head(1).index.values[0]])
    decay_rate0=gp0[1]
    gp1=derive_growth_params(slice)
    decay_rate1=gp1[1]

    projection=df.tail(8).head(1)["7-day-forward-projection-total"].values[0]
    today=round(df.tail(1)["total"].values[0])
    error=projection-today
    relative_error=(projection-today)/today*100

    summary="""
    <style>
        span.good {
            color: green;
        }
        span.bad {
            color: red;
        }
    </style>
    """+f"""
    <h1>Weekly Summary</h1>
    <br/>
    <p>Delta values are with respect to values from 1 week prior.</p>
    </p>
    <pre>
    Date: {df.tail(1)['date'].values[0]} (#{df.tail(1).index.values[0]})

    New Cases Reported Today: {round(df.tail(1)["total"].values[0])} {g(delta["total"].values[0])}
    Cumulative (since 2021-06-16): {round(df.tail(1)["cumulative"].values[0])} {f(delta["cumulative"].values[0])}
    Corrections (since 2021-06-16): {round(df.tail(1)["correction"].values[0])} {f(delta["correction"].values[0])}

    Projection (from 7 days ago): {round(df.tail(8).head(1)["7-day-forward-projection-total"].values[0])}
    Projection Error: {round(error)} ({hh(round(relative_error,1))}%)
    Projection (for next week): {round(df.tail(1)["7-day-forward-projection-total"].values[0])}

    Cumulative Growth Rate: {round(df.tail(1)["ols-growth-rate"].values[0],2)}% per day {g(delta["ols-growth-rate"].values[0])}
    Linear Growth Rate : {round(df.tail(1)["linear-growth-rate"].values[0],2)}% per day {g(delta["linear-growth-rate"].values[0])}
    Reff: {round(df.tail(1)["Reff"].values[0],2)} {g(delta["Reff"].values[0])}
    Doubling Period:  {round(df.tail(1)["doubling-period"].values[0],1)} days {h(delta["doubling-period"].values[0])}

    Growth Decay Rate: {round(decay_rate1, 2)}% per day {g(round(decay_rate1 - decay_rate0, 2))}
    ln-ln Gradient: {round(df.tail(1)["ltlc-gradient"].values[0], 3)} {g(round(delta["ltlc-gradient"].values[0], 3))}
    </pre>"""

    return summary


def plot_linear_growth(df):
    ax=df[["ols-growth-rate", 'linear-growth-rate', "linear-growth-rate-max", "linear-growth-rate-min", "linear-growth-rate-mean"]][df["linear-growth-rate-max"]<40].plot(figsize=(10,10))
    ax.grid()
    return ax

def plot_linear_growth_error(df):
    ax=(df["linear-growth-rate-relative-error"]).rolling(window=5).max().plot(figsize=(10,10))
    ax.plot((df["linear-growth-rate-relative-error"]).rolling(window=5).min())
    ax.plot(df["linear-growth-rate-relative-error"])
    ax.plot(df["linear-growth-rate-relative-error"].cumsum()/(df.index+1))
    ax.grid()
    return ax

def decay_rate_estimates(df, min_period=7, max_period=7, step=1):
    W,s=0,None
    for N in range(min_period,max_period+1,step):
        shifted_df=df.shift(N)
        p=((df["ols-growth-rate"]/shifted_df["ols-growth-rate"])**(1/N)-1)*100
        w=N
        s=s+(p*w) if s is not None else p
        W += w
        yield (N,p)
    S=s/W
    yield (None,S)

def plot_decay_rate_estimates(df, outbreak="Sydney 2021", step=1):
    legends=[]

    estimates=[r for r in decay_rate_estimates(df, 7, 28)]
    _,S=estimates[-1]
    for e in estimates[0:-1]:
        e[1].plot(legend=f"period={e[0]}")

    ax = S.plot(color="black", linewidth=3, figsize=(10,10))

    ax.legend(legends)
    ax.grid()
    ax.set_title(f"Decay Rate Estimates - {outbreak}")
    ax.set_xlabel("Day of Outbreak")
    ax.set_ylabel("Growth Decay Rate (%)")
    ax.set_xlim(left=0)

    return ax

def peak_cases(df, g):
    s=select_outbreak(project_ols_growth_rate_min(df, 365-len(df), g, 'ols-growth-rate'))
    max_s = s[s["total"] == s["total"].max()][["date", "total"]]
    return {
        "day": int(max_s.index.values[0]),
        "date": max_s["date"].values[0],
        "decay_rate": round(g,3),
        "total": int(max_s["total"].values[0]),
    }


def peak_cases_projection(df):
    out=[r for r in decay_rate_estimates(df, 7, 28)]
    S=out[-1][1]

    decay_rates={
        "mean": S.mean(),
        "last": S.tail(1).values[0]
    }

    if S.mean() < 0 and S.tail(1).values[0] < 0:
        decay_rates["mid"] = -np.sqrt(S.mean()*S.tail(1).values[0])

    decay_rates={ k: decay_rates[k] for k in decay_rates if decay_rates[k] <= 0 }

    out=OrderedDict()

    for k in [t[0] for t in sorted([(k,decay_rates[k]) for k in decay_rates], key=lambda t: t[1])]:
        out[k]=peak_cases(df, decay_rates[k])
    return out

def plot_new_cases_projection(df):
    out=[r for r in decay_rate_estimates(df, 7, 28)]
    S=out[-1][1]

    results={
        "min": S.min(),
        "max": S.max(),
        "mean": S.mean(),
        "last": S.tail(1).values[0],
        "mid": -np.sqrt(S.mean()*S.tail(1).values[0])
    }

    results={ k: results[k] for k in results if results[k] <= 0 }

    legends=["daily new cases"]
    ax=df['total'].plot()
    for k in results:
        v=results[k]
        s=select_outbreak(project_ols_growth_rate_min(df, 365-len(df), v, 'ols-growth-rate'))
        _max_total=s['total'].max()
        _max_date=s.loc[s["total"]==_max_total, "date"].values[0]
        legends.append(f"{k} decay rate={round(v,2)}%, max={int(round(_max_total,0))}, date={_max_date}")
        style="dashed" if k == "mid" else "dotted"
        ax=s[s.index>len(df.index)]['total'].plot(figsize=(10,10),linestyle=style)
    ax.legend(legends)
    ax.grid()
    ax.set_ylim(top=50000)
    ax.set_ybound(1, 50000)
    ax.set_yscale('log')
    ax.set_xticks([r for r in range(0,365, 14)])
    #ax.set_yticks([r for r in range(0,3500, 100)])
    ax.set_title(f"Projected Daily New Cases For Various Observed Decay Rates ({df.tail(1)['date'].values[0]})")
    return ax

def animate_new_cases_plot(df, days, fn):
    images=[]
    for i in range(0, days):
        ax=plot_new_cases_projection(df.head(len(df)-days+1+i))
        b=BytesIO()
        ax.figure.savefig(b, format="png")
        plt.show()
        images.append(Image.open(b))

    images[0].save(fn, save_all=True, append_images=images[1:], loop=0, duration=1500)

def plot_health_outcomes(df):
    window=7
    endog=df['icu']
    exog=sm.add_constant(df['cumulative'])
    model = RollingOLS(endog=endog, exog=exog, window=window, min_nobs=window)
    icu_params=model.fit().params
    icu_m,icu_b=(icu_params['cumulative'].tail(1).values[0], icu_params['const'].tail(1).values[0])

    icu_m_7,icu_b_7=(icu_params['cumulative'].head(len(icu_params)-7).tail(1).values[0], icu_params['const'].head(len(icu_params)-7).tail(1).values[0])

    endog=df['hospitalised']
    exog=sm.add_constant(df['cumulative'])
    model = RollingOLS(endog=endog, exog=exog, window=window, min_nobs=window)
    hospitalised_params=model.fit().params
    hospitalised_m,hospitalised_b=(hospitalised_params['cumulative'].tail(1).values[0], hospitalised_params['const'].tail(1).values[0])
    hospitalised_m_7,hospitalised_b_7=(hospitalised_params['cumulative'].head(len(hospitalised_params)-7).tail(1).values[0], hospitalised_params['const'].head(len(hospitalised_params)-7).tail(1).values[0])

    slice=df.loc[df['deaths'].notna(), ['deaths', 'cumulative']]
    endog=slice['deaths']
    exog=sm.add_constant(slice['cumulative'])
    model = RollingOLS(endog=endog, exog=exog, window=window, min_nobs=window)
    deaths_params=model.fit().params
    deaths_m,deaths_b=(deaths_params['cumulative'].tail(1).values[0], deaths_params['const'].tail(1).values[0])
    deaths_m_7,deaths_b_7=(deaths_params['cumulative'].head(len(deaths_params)-7).tail(1).values[0], deaths_params['const'].head(len(deaths_params)-7).tail(1).values[0])

    ax=df.plot(y='hospitalised', x='cumulative', figsize=(10,10))
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.plot(df['cumulative'], df['cumulative']*hospitalised_m_7+hospitalised_b_7, linestyle="dotted", color="C0")
    ax.plot(df['cumulative'], df['cumulative']*hospitalised_m+hospitalised_b, linestyle="dashed", color="C0")
    ax.set_title("Concurrent Health Outcomes vs Cumulative Cases")
    ax.plot(df['cumulative'], df['icu'], color="C1")
    ax.plot(df['cumulative'], df['cumulative']*icu_m_7+icu_b_7, linestyle="dotted", color="C1")
    ax.plot(df['cumulative'], df['cumulative']*icu_m+icu_b, linestyle="dashed", color="C1")
    ax.plot(df['cumulative'], df['deaths'], color="C2")
    ax.plot(df['cumulative'], df['cumulative']*deaths_m_7+deaths_b_7, linestyle="dotted", color="C2")
    ax.plot(df['cumulative'], df['cumulative']*deaths_m+deaths_b, linestyle="dashed", color="C2")
    percentage=lambda x,y : f"{round(x*100,y)}%"
    ax.legend([
        "hospitalised",
        f"hospitalised (7-day fit, last week), m={percentage(hospitalised_m_7,1)}",
        f"hospitalised (7-day fit), m={percentage(hospitalised_m,1)}",
        "icu",
        f"icu (7-day fit, last week), m={percentage(icu_m_7,2)}",
        f"icu (7-day fit), m={percentage(icu_m,2)}",
        "deaths",
        f"deaths (7-day fit, last week), m={percentage(deaths_m_7,2)}",
        f"deaths (7-day fit), m={percentage(deaths_m,2)}"
    ])
    return ax
