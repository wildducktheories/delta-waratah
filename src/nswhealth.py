import json
import requests
import re
import os
import pandas as pd
pd.set_option('display.max_rows', None)

files=[
  "20210617_00.aspx",
  "20210618_00.aspx",
  "20210619_00.aspx",
  "20210620_00.aspx",
  "20210621_00.aspx",
  "20210622_00.aspx",
  "20210623_00.aspx",
  "20210624_00.aspx",
  "20210625_01.aspx",
  "20210626_00.aspx",
  "20210627_00.aspx",
  "20210628_00.aspx",
  "20210629_01.aspx",
  "20210630_00.aspx",
  "20210701_00.aspx",
  "20210702_00.aspx",
  "20210703_01.aspx",
  "20210704_00.aspx",
  "20210705_00.aspx",
  "20210706_00.aspx",
  "20210707_00.aspx",
  "20210708_00.aspx",
  "20210709_00.aspx",
  "20210710_00.aspx",
  "20210711_00.aspx",
  "20210712_00.aspx",
  "20210713_00.aspx",
  "20210714_00.aspx",
  "20210715_00.aspx",
  "20210716_00.aspx",
  "20210717_00.aspx",
  "20210718_00.aspx",
  "20210719_00.aspx",
  "20210720_00.aspx",
  "20210721_00.aspx",
  "20210722_00.aspx",
  "20210723_00.aspx",
  "20210724_00.aspx",
  "20210725_00.aspx",
  "20210726_00.aspx",
  "20210727_00.aspx",
  "20210728_00.aspx",
  "20210729_00.aspx",
  "20210730_00.aspx",
  "20210731_01.aspx",
  "20210801_00.aspx",
  "20210802_00.aspx",
  "20210803_00.aspx",
  "20210804_01.aspx",
  "20210805_00.aspx",
  "20210806_00.aspx",
  "20210807_00.aspx",
  "20210808_00.aspx",
  "20210809_00.aspx",
  "20210810_00.aspx",
  "20210811_00.aspx",
  "20210812_00.aspx",
  "20210813_02.aspx",
  "20210814_00.aspx",
  "20210815_00.aspx",
  "20210816_01.aspx",
  "20210817_00.aspx",
  "20210818_01.aspx",
  "20210819_00.aspx",
  "20210820_01.aspx",
  "20210821_01.aspx"
]

def fn_to_date(fn):
    parse_fn=re.compile("(\d\d\d\d)(\d\d)(\d\d).*")
    (yyyy,mm,dd)=parse_fn.match(fn).groups()
    date=f"{yyyy}-{mm}-{dd}"
    return date

date_dict={fn_to_date(f): f for f in files}

def load_statistics(date):
    fn=f"archive/{date}/statistics.html"
    if not os.path.exists(fn):
        file=date_dict[date]
        response = requests.get(f"https://www.health.nsw.gov.au/news/Pages/{file}")
        response.raise_for_status()
        content=response.content
        os.makedirs(f"archive/{date}", exist_ok=True)
        with open(fn, "wb") as f:
            f.write(content)
    else:
        with open(fn, "rb") as f:
            content=f.read()
    return content.decode('utf-8')

def extract_total(date, debug=False):
    content=load_statistics(date)
    content=content.replace("​", "")
    content=content.replace("two", "2")
    content=content.replace("five", "5")
    content=content.replace("no", "0")
    pattern_cumulative=re.compile("^.*There have been ([\d,]*) locally acquired cases reported since 16 June 2021")
    pattern=re.compile("^.*<p[^>]*>[. ]*NSW recorded ([^ ]*) .*locally acquired cases of COVID-19")
    pattern_partial=re.compile("^.*NSW recorded")
    total=None
    cumulative=None
    for c in content.split("\n"):
        g=pattern_cumulative.match(c)
        if g:
            cumulative=int(g.groups()[0].replace(",",""))
        if pattern_partial.match(c):
            g=pattern.match(c)
            if g:
                total=int(g.groups()[0])
                continue
            else: 
                if debug:
                    print(c)
    return (date, total, cumulative)

def load_nswhealth_stats():
    out=[]
    for date in date_dict:
        (date, total, cumulative)=extract_total(date)
        #if (total is None or cumulative is None):
        out.append((date, total, cumulative))
    df = pd.DataFrame(columns=['date', 'total', 'cumulative'], data=out)
    df.loc[df["cumulative"].isna(), 'cumulative'] = df.loc[df["cumulative"].isna(), 'total'].cumsum()
    return df

def calculate_differences(df):
    df['cumulative_calc'] = df['total'].cumsum()
    df['total_calc']=df['cumulative']-df['cumulative'].shift(1)
    df['cumulative_diff']=df['cumulative_calc']-df['cumulative']
    df['total_diff']=df['total_calc']-df['total']
    return df