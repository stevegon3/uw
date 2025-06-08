import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_str="""Customer Number	Purchase State	Dollars Spent	Customer Satisfaction level
    1	Georgia	$1,511	82
    2	Georgia	$1,489	90
    4	Georgia	$100	43
    9	Georgia	$786	99
    12	Georgia	$659	85
    13	Georgia	$5,853	97
    15	Georgia	$572	71
    17	Georgia	$528	68
    3	Florida	$1,015	85
    5	Florida	$959	63
    6	Florida	$915	73
    7	Florida	$885	84
    8	Florida	$795	71
    10	Florida	$695	71
    11	Florida	$680	51
    14	Florida	$579	56
    16	Florida	$551	51"""
    columns = data_str.split('\n')[0].split('\t')
    data = []
    for r in data_str.split('\n')[1:]:
        cust, st, spend, sat = r.split('\t')
        spend = float(spend.replace('$','').replace(',',''))
        sat = int(sat)
        print(cust, st, spend, sat)
        data.append([cust, st, spend, sat])
    df = pd.DataFrame(data=data, columns=columns)
    df['Customer Satisfaction level'] = pd.to_numeric(df['Customer Satisfaction level'], errors='coerce')
    df.drop()