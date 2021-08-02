import numpy as np
import pandas as pd

from .functions import calc_cumulative, calculate_ltlc_params

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
    calculate_ltlc_params(df),
    return df

def edit_dates(df):
    newyear=df['date'][df['date']=='01/01'].index.values
    if len(newyear) > 0:
        df.loc[df.index < newyear[0],['date']]=df['date'].loc[df.index < newyear[0]]+'/20'
    else:
        newyear=[0]
    df.loc[df.index >= newyear[0],['date']]=df['date'].loc[df.index >= newyear[0]]+'/21'
    df['date']=df['date'].apply(rewrite_date)
    return df

def rewrite_date(d):
    p = d.split('/')
    return '-'.join(['20'+p[2], p[1], p[0]])


