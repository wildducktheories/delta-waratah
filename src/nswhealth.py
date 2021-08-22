import json
import requests
import re
import os
import pandas as pd

from collections import OrderedDict

pd.set_option('display.max_rows', None)

def fn_to_date(fn):
    parse_fn=re.compile("(\d\d\d\d)(\d\d)(\d\d).*")
    (yyyy,mm,dd)=parse_fn.match(fn).groups()
    date=f"{yyyy}-{mm}-{dd}"
    return date

DATE_DICT=None

def load_datedict():
    global DATE_DICT
    if not DATE_DICT:
        response=requests.get("https://www.health.nsw.gov.au/news/Pages/2021-nsw-health.aspx")
        response.raise_for_status()
        content=response.content
        pattern=re.compile('.*<a href="([0-9_]*\\.aspx)" xmlns:ddwrt="http://schemas.microsoft.com/WebParts/v2/DataView/runtime">COVID-19 \\(Coronavirus\\) .*</a>')
        out=[]
        for line in content.decode('utf-8').split("\n"):
            g=pattern.match(line)
            if g:
                out.append(g.groups()[0])
            else:
                pass
        out=sorted(out)
        out=filter(lambda x: x > '20210616', out)
        DATE_DICT=OrderedDict([(fn_to_date(k), k) for k in out])
    return DATE_DICT

def load_statistics(date):
    fn=f"archive/{date}/statistics.html"
    if not os.path.exists(fn):
        file=load_datedict()[date]
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
    for date in load_datedict():
        (date, total, cumulative)=extract_total(date)
        #if (total is None or cumulative is None):
        out.append((date, total, cumulative))
    df = pd.DataFrame(columns=['date', 'total', 'cumulative_corrected'], data=out)
    df['cumulative'] = df['total'].cumsum()
    df['correction'] = df['cumulative_corrected'] - df['cumulative']
#    df.loc[df["cumulative"].isna(), 'cumulative'] = df.loc[df["cumulative"].isna(), 'total'].cumsum()
    return df
