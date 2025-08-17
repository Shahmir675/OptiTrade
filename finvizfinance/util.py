"""
.. module:: util
   :synopsis: General function for the package.

.. moduleauthor:: Tianning Li <ltianningli@gmail.com>
"""

import sys
from datetime import datetime

import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) \
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
}
session = requests.Session()

proxy_dict = None


def set_proxy(proxies):
    global proxy_dict
    proxy_dict = proxies


def web_scrap(url, params=None):
    """Scrap website.

    Args:
        url(str): website
        params(dict): request parameters
    Returns:
        soup(beautiful soup): website html
    """
    try:
        website = session.get(
            url, params=params, headers=headers, timeout=10, proxies=proxy_dict
        )
        website.raise_for_status()
        soup = BeautifulSoup(website.text, "lxml")
    except requests.exceptions.HTTPError as err:
        raise Exception(err)
    except requests.exceptions.Timeout as err:
        raise Exception(err)
    return soup


def image_scrap(url, ticker, out_dir):
    """scrap website and download image

    Args:
        url(str): website (image)
        ticker(str): output image name
        out_dir(str): output directory
    """
    try:
        r = session.get(
            url, stream=True, headers=headers, timeout=10, proxies=proxy_dict
        )
        r.raise_for_status()
        r.raw.decode_content = True
        if len(out_dir) != 0:
            out_dir += "/"
        f = open("{}{}.jpg".format(out_dir, ticker), "wb")
        f.write(r.content)
        f.close()
    except requests.exceptions.HTTPError as err:
        raise Exception(err)
    except requests.exceptions.Timeout as err:
        raise Exception(err)


def scrap_function(url):
    """Scrap forex, crypto information.

    Args:
        url(str): website
    Returns:
        df(pandas.DataFrame): performance table
    """
    soup = web_scrap(url)
    table = soup.find("table", class_="groups_table")
    rows = table.find_all("tr")
    table_header = [i.text.strip() for i in rows[0].find_all("th")][1:]
    frame = []
    rows = rows[1:]
    num_col_index = [i for i in range(2, len(table_header))]
    for row in rows:
        cols = row.find_all("td")[1:]
        info_dict = {}
        for i, col in enumerate(cols):
            if i not in num_col_index:
                info_dict[table_header[i]] = col.text
            else:
                info_dict[table_header[i]] = number_covert(col.text)
        frame.append(info_dict)
    return pd.DataFrame(frame)


def image_scrap_function(url, chart, timeframe, urlonly):
    """Scrap forex, crypto information.

    Args:
        url(str): website
        chart(str): choice of chart
        timeframe (str): choice of timeframe(5M, H, D, W, M)
        urlonly (boolean):  choice of downloading chart
    """
    if timeframe == "5M":
        url += "m5"
    elif timeframe == "H":
        url += "h1"
    elif timeframe == "D":
        url += "d1"
    elif timeframe == "W":
        url += "w1"
    elif timeframe == "M":
        url += "mo"
    else:
        raise ValueError("Invalid timeframe.")

    soup = web_scrap(url)
    content = soup.find("div", class_="container")
    imgs = content.find_all("img")
    for img in imgs:
        website = img["src"]
        name = website.split("?")[1].split("&")[0].split(".")[0]
        chart_name = name.split("_")[0]
        if chart.lower() == chart_name:
            charturl = "https://finviz.com/" + website
            if not urlonly:
                image_scrap(charturl, name, "")
            return charturl
        else:
            continue


def number_covert(num):
    """Convert number(str) to number(float)

    Args:
        num(str): number as a string
    Return:
        num(float or None): number converted to float or None
    """
    if not num or num == "-":  # Check if the string is empty or is "-"
        return None
    num = num.strip()  # Remove any surrounding whitespace
    if num[-1] == "%":
        return float(num[:-1]) / 100
    elif num[-1] == "B":
        return float(num[:-1]) * 1000000000
    elif num[-1] == "M":
        return float(num[:-1]) * 1000000
    elif num[-1] == "K":
        return float(num[:-1]) * 1000
    else:
        return float(num.replace(",", ""))


def format_datetime(date_str):
    time_str = date_str.split(" ")[-1]
    time_in_24hr = datetime.strptime(time_str, "%I:%M%p").strftime("%H:%M")

    if date_str.lower().startswith("today"):
        us_timezone = pytz.timezone("US/Eastern")
        today = datetime.now(us_timezone)

        hour, minute = map(int, time_in_24hr.split(":"))

        result = datetime(today.year, today.month, today.day, hour, minute)
    else:
        result = datetime.strptime(date_str, "%b-%d-%y %I:%M%p")

    us_timezone = pytz.timezone("US/Eastern")
    result = us_timezone.localize(result)

    pkt_timezone = pytz.timezone("Asia/Karachi")
    result_in_pkt = result.astimezone(pkt_timezone)

    return result_in_pkt.strftime("%Y-%m-%d %H:%M:%S")


def progress_bar(page, total):
    bar_len = 30
    filled_len = int(round(bar_len * page / float(total)))
    bar = "#" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write("[Info] loading page [{}] {}/{} \r".format(bar, page, total))
    sys.stdout.flush()
