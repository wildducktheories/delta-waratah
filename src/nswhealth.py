import json
import requests
import re
import os
import numpy as np
import pandas as pd

from collections import OrderedDict
from .functions import add_days

pd.set_option('display.max_rows', None)

def fn_to_date(fn):
    parse_fn=re.compile("(\d\d\d\d)(\d\d)(\d\d).*")
    (yyyy,mm,dd)=parse_fn.match(fn).groups()
    date=f"{yyyy}-{mm}-{dd}"
    return date

def load_index():
    response=requests.get("https://www.health.nsw.gov.au/news/Pages/2021-nsw-health.aspx")
    response.raise_for_status()
    content=response.content
    pattern=re.compile('.*<a href="([0-9_]*\\.aspx)" xmlns:ddwrt="http://schemas.microsoft.com/WebParts/v2/DataView/runtime">COVID-19 \\(Coronavirus\\) .*</a>')
    pattern2=re.compile('.*<a href="([0-9_]*\\.aspx)" xmlns:ddwrt="http://schemas.microsoft.com/WebParts/v2/DataView/runtime">Coronavirus \\(COVID-19\\) statistics.*</a>')
    out=[]
    for line in content.decode('utf-8').split("\n"):
        g=pattern.match(line)
        g2=pattern2.match(line)
        if g:
            out.append(g.groups()[0])
        elif g2:
            out.append(g2.groups()[0])
        else:
            pass
    out=sorted(out)
    out=filter(lambda x: x > '20210616', out)
    return OrderedDict([(fn_to_date(k), k) for k in out])

def load_statistics(file, date):
    fn=f"archive/{date}/statistics.html"
    if not os.path.exists(fn):
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

def parse_statistics_html(html, debug=False):

    html=html.replace("​", "")
    html=html.replace("two", "2")
    html=html.replace("three", "3")
    html=html.replace("four", "4")
    html=html.replace("Four", "4")
    html=html.replace("five", "5")
    html=html.replace("six", "6")
    html=html.replace("seven", "7")
    html=html.replace(" no ", " 0 ")
    pattern_obsfucate=re.compile("^(.*)<span[^>]*>(.*)</span>(.*)")
    pattern_cumulative=re.compile("^.*There have been ([\d,]*) locally acquired cases reported since 16 June 2021")
    pattern=re.compile("^.*NSW recorded ([^ ]*) .*locally acquired cases of COVID-19")
    pattern_2=re.compile("^.*NSW recorded ([\d,]*) new cases of COVID-19 in the 24 hours to 8pm last night.")

    pattern_partial=re.compile("^.*NSW recorded")
    pattern_health=re.compile(".*There are currently ([\d,]*) COVID-19 cases admitted to hospital, with ([\d,]*) people in intensive care, (.*) of whom require ventilation.")
    pattern_deaths=re.compile(".*There have been ([\d,]*) COVID.*related deaths")
    pattern_deaths_2=re.compile(".*This brings the number of COVID-related deaths to ([\d,]*)")
    pattern_deaths_3=re.compile(".*This brings to ([\d,]*) the number of COVID-related deaths")
    pattern_deaths_4=re.compile(".*The number of COVID-related deaths during the current outbreak is now ([\d,]*)")
    pattern_deaths_5=re.compile(".*NSW has recorded ([\d,]*) COVID-19 related deaths since 16 June 2021")
    pattern_deaths_6=re.compile(".*Deaths \\(in NSW from confirmed cases\\).*<td[^>]*> *([\d,]+).*Total tests")
    total=None
    cumulative=None
    hospitalised=None
    icu=None
    ventilated=None
    deaths=None

    for c in html.split("\n"):
        g=pattern_obsfucate.match(c)
        while g is not None:
            c="".join(g.groups())
            g=pattern_obsfucate.match(c)

        g=pattern_cumulative.match(c)
        if g:
            cumulative=int(g.groups()[0].replace(",",""))
        if pattern_partial.match(c):
            g=pattern.match(c)
            if g:
                total=int(g.groups()[0].replace(",",""))
            else: 
                g = pattern_2.match(c)
                if g:
                    total=int(g.groups()[0].replace(",",""))
                else:
                    if debug:
                        print(c)
        g=pattern_health.match(c)
        if g:
            hospitalised,icu,ventilated=[int(v.replace(',', '')) for v in g.groups()]

        g=pattern_deaths.match(c)
        if g:
            deaths=int(g.groups()[0].replace(",", ""))

        g=pattern_deaths_2.match(c)
        if g:
            deaths=int(g.groups()[0].replace(",", ""))

        g=pattern_deaths_3.match(c)
        if g:
            deaths=int(g.groups()[0].replace(",", ""))

        g=pattern_deaths_4.match(c)
        if g:
            deaths=int(g.groups()[0].replace(",", ""))

        g=pattern_deaths_5.match(c)
        if g:
            deaths=int(g.groups()[0].replace(",", ""))

        g=pattern_deaths_6.match(c)
        if g:
            # print(g.groups()[0], int(g.groups()[0]))
            deaths=int(g.groups()[0].replace(",", ""))-56 # total deaths - 56

    return (total, cumulative, hospitalised, icu, ventilated, deaths)

def load_statistics_json(date, stats):
    date=stats.get("date", date)
    total=stats.get("total")
    cumulative=stats.get("cumulative_corrected")
    hospitalised=stats.get("hospitalised")
    icu=stats.get("icu")
    ventilated=stats.get("ventilated")
    deaths=stats.get("deaths")
    if (type(icu) == tuple) or (icu is None):
        icu=float("NaN")
    return (date, total, cumulative, hospitalised, icu, ventilated, deaths)

def load_nswhealth_stats(limit_date):
    out=[]
    index=load_index()
    for date in index:
        (total, cumulative, hospitalised, icu, ventilated, deaths)=parse_statistics_html(load_statistics(index[date], date))
        json_file=f'archive/{date}/statistics.json'
        if total is None or cumulative is None:
            if not os.path.exists(json_file):
                with open(json_file, "wb") as f:
                    f.write(json.dumps({
                        "date": date,
                        "total": total,
                        "cumulative_corrected": cumulative,
                        "hospitalised": hospitalised,
                        "icu": icu,
                        "ventilated": ventilated,
                        "deaths": deaths
                    }).encode('utf-8'))

        if date <= limit_date:
            out.append((date, total, cumulative, hospitalised, icu, ventilated, deaths))

    dates=OrderedDict([(t[0],t) for t in out])

    # scan all dates and override any missing with statistics.json
    date='2021-06-16'
    while date <= limit_date:
        json_file=f'archive/{date}/statistics.json'
        if os.path.exists(json_file):
            with open(json_file, "rb") as f:
                tmp = load_statistics_json(date, json.loads(f.read().decode("utf-8")))
                dates[date] = tmp

        date = add_days(date, 1)

    df = pd.DataFrame(columns=['date', 'total',  'cumulative_corrected', 'hospitalised', 'icu', 'ventilated', 'deaths'], data=dates.values())
    df['cumulative'] = df['total'].cumsum()
    df['correction'] = df['cumulative_corrected'] - df['cumulative']
    return df
