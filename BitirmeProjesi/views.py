import locale
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.linear_model import LinearRegression


def get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='utf-8')


def mainscreen(request):
    data = get_data('BitirmeProjesi/data.csv')

    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')

    sort_columns = {
        '1': ('MAL_ADI', True),
        '2': ('MAL_ADI', False),
        '3': ('ORTALAMA_UCRET', True),
        '4': ('ORTALAMA_UCRET', False)
    }

    filter_columns = {
        '1': ('MAL_TURU', 'MEYVE'),
        '2': ('MAL_TURU', 'SEBZE'),
        '3': ('MAL_TURU', 'İTHAL..'),
    }

    def process_data(data: pd.DataFrame) -> pd.DataFrame:
        data['TARIH'] = pd.to_datetime(data['TARIH'])
        return data.groupby('MAL_ADI').first().reset_index()[['MAL_ADI', 'ORTALAMA_UCRET', 'MAL_TURU']]

    def sort_data(data: pd.DataFrame, sort_value: str) -> pd.DataFrame:
        column, ascending = sort_columns.get(sort_value, (None, None))
        if column:
            if column == 'MAL_ADI':
                return data.sort_values(by=column, key=lambda x: x.str.normalize('NFKD'), ascending=ascending)
            else:
                return data.sort_values(by=column, ascending=ascending)
        return data

    def get_sorted_data(data: pd.DataFrame, sort_value: str) -> pd.DataFrame:
        processed_data = process_data(data)
        return sort_data(processed_data, sort_value)

    def get_filtered_data(data: pd.DataFrame, filter_value: str) -> pd.DataFrame:
        processed_data = process_data(data)
        return filter_data(processed_data, filter_value)

    def filter_data(data: pd.DataFrame, filter_value: str) -> pd.DataFrame:
        column, tur = filter_columns.get(filter_value, (None, None))
        if column:
            return data[data[column] == tur]
        return data

    if request.method == "POST":
        sort_value = request.POST.get('sort_value')
        filter_value = request.POST.get('filter_value')

        if sort_value:
            return render(request, 'mainscreen.html', {"datas": get_sorted_data(data, sort_value).to_dict('records')})
        if filter_value:
            return render(request, 'mainscreen.html',
                          {"datas": get_filtered_data(data, filter_value).to_dict('records')})

    elif request.method == "GET":
        return render(request, 'mainscreen.html', {"datas": process_data(data).to_dict('records')})


def detailview(request, data_haftalık, data_aylık, data_yıllık, maximum, minimum):
    return render(request, 'detailpage.html',
                  {'data_haftalık': data_haftalık, 'data_aylık': data_aylık, 'data_yıllık': data_yıllık,
                   'maxi': maximum, 'mini': minimum})


def csv_detail(request, urun_adı):
    df = get_data('BitirmeProjesi/data.csv')

    product_data = df[df['MAL_ADI'] == urun_adı]
    product_data['TARIH'] = pd.to_datetime(product_data['TARIH'])

    grouped_aylık = product_data.groupby(['MAL_ADI', product_data['TARIH'].dt.to_period('M')])[
        'ORTALAMA_UCRET'].mean().reset_index().sort_values(by='TARIH', ascending=True)
    grouped_yıllık = product_data.groupby(['MAL_ADI', product_data['TARIH'].dt.to_period('Y')])[
        'ORTALAMA_UCRET'].mean().reset_index().sort_values(by='TARIH', ascending=True)
    grouped_haftalık = (
        product_data.assign(TARIH_week=product_data['TARIH'].dt.to_period('W')).groupby(['MAL_ADI', 'TARIH_week'])[
            'ORTALAMA_UCRET'].mean().reset_index().sort_values(by='TARIH_week', ascending=True)
    )
    grouped_gunluk = product_data.groupby(['MAL_ADI', product_data['TARIH'].dt.to_period('D')])[
        'ORTALAMA_UCRET'].mean().reset_index().sort_values(by='TARIH', ascending=True)

    week = grouped_haftalık['TARIH_week'].max()
    start_week_period = week.start_time.to_period('D')
    end_week_period = week.end_time.to_period('D')

    filtered_data = grouped_gunluk[
        (grouped_gunluk['TARIH'] >= start_week_period) & (grouped_gunluk['TARIH'] <= end_week_period)]

    data_aylık = [{'MAL_ADI': row['MAL_ADI'], 'ORTALAMA_UCRET': f"{row['ORTALAMA_UCRET']:.2f}"} for index, row in
                  grouped_aylık.iterrows()]
    data_yıllık = [{'MAL_ADI': row['MAL_ADI'], 'TARIH': row['TARIH'], 'ORTALAMA_UCRET': f"{row['ORTALAMA_UCRET']:.2f}"} for index, row in
                   grouped_yıllık.iterrows()]
    data_haftalık = [{'MAL_ADI': row['MAL_ADI'], 'ORTALAMA_UCRET': f"{row['ORTALAMA_UCRET']:.2f}"} for index, row in
                     grouped_haftalık.iterrows()]
    data_filtered = [{'MAL_ADI': row['MAL_ADI'], 'ORTALAMA_UCRET': f"{row['ORTALAMA_UCRET']:.2f}"} for index, row in
                     filtered_data.iterrows()]
    data_date = [{'MAL_ADI': row['MAL_ADI'], 'TARIH': row['TARIH'], 'ORTALAMA_UCRET': f"{row['ORTALAMA_UCRET']:.2f}"}
                 for index, row in
                 grouped_gunluk.iterrows()]

    max_min_df = pd.DataFrame(data_haftalık)
    max_min_df['ORTALAMA_UCRET'] = max_min_df['ORTALAMA_UCRET'].astype(float)
    maximum = max_min_df['ORTALAMA_UCRET'].max()
    minimum = max_min_df['ORTALAMA_UCRET'].min()

    context = {
        'data_haftalık': data_haftalık,
        'data_aylık': data_aylık,
        'data_yıllık': data_yıllık,
        'maxi': maximum,
        'mini': minimum,
        'filtered': data_filtered,
        'date_select': data_date,
        'predictions': predict(urun_adı),
    }

    return render(request, 'detailpage.html', context)


def filter_data_by_search_term(data: pd.DataFrame, search_term: str) -> pd.DataFrame:
    search_terms = [term.lower() for term in search_term.split()]
    search_str = ''.join(search_terms)
    return data[data['MAL_ADI'].str.contains(search_str, case=False)]


def search(request):
    def process_data(data: pd.DataFrame) -> pd.DataFrame:
        data['TARIH'] = pd.to_datetime(data['TARIH'])
        return data.groupby('MAL_ADI').first().reset_index()[['MAL_ADI', 'ORTALAMA_UCRET', 'MAL_TURU']]

    try:
        search_term = request.GET.get('search')
        if not search_term:
            data = get_data('BitirmeProjesi/data.csv')
            return render(request, 'mainscreen.html', {'datas': process_data(data).to_dict('records')})
        data = get_data('BitirmeProjesi/data.csv')
        filtered_data = filter_data_by_search_term(data, search_term)
        return render(request, 'mainscreen.html', {'datas': process_data(filtered_data).to_dict('records')})
    except (AttributeError, ValueError) as e:
        return {"error": str(e)}


def notification(request):
    data = get_data('BitirmeProjesi/data.csv')
    data['TARIH'] = pd.to_datetime(data['TARIH'])

    data_grouped = data.sort_values('TARIH', ascending=False).groupby('MAL_ADI').head(3).reset_index(drop=True)

    azalan_mallar = []
    yukselen_mallar = []

    for name, group in data_grouped.groupby('MAL_ADI'):
        if group.iloc[0]['ORTALAMA_UCRET'] > group.iloc[-1]['ORTALAMA_UCRET']:
            yukselen_mallar.append({
                'MAL_ADI': name,
                'ORTALAMA_UCRET_latest': group.iloc[0]['ORTALAMA_UCRET'],
                'ORTALAMA_UCRET_previous': group.iloc[-1]['ORTALAMA_UCRET'],
                'MAL_TURU': group.iloc[0]['MAL_TURU']
            })

    for name, group in data_grouped.groupby('MAL_ADI'):
        if group.iloc[0]['ORTALAMA_UCRET'] < group.iloc[-1]['ORTALAMA_UCRET']:
            azalan_mallar.append({
                'MAL_ADI': name,
                'ORTALAMA_UCRET_latest': group.iloc[0]['ORTALAMA_UCRET'],
                'ORTALAMA_UCRET_previous': group.iloc[-1]['ORTALAMA_UCRET'],
                'MAL_TURU': group.iloc[0]['MAL_TURU']
            })

    return render(request, 'notification.html', {"data_azalan": azalan_mallar, "data_yukselen": yukselen_mallar})


def predict(mal_adi):
    collection = get_data('BitirmeProjesi/data.csv')

    df = collection[
        collection['MAL_ADI'] == mal_adi].copy()
    df['TARIH'] = pd.to_datetime(df['TARIH'])
    df.set_index('TARIH', inplace=True)

    model = LinearRegression()
    x = df.index.map(datetime.timestamp).values.reshape(-1, 1)
    y = df['ORTALAMA_UCRET'].values
    model.fit(x, y)

    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 31)]
    future_dates_sec = np.array([datetime.timestamp(date) for date in future_dates]).reshape(-1, 1)
    future_predictions = model.predict(future_dates_sec)

    formatted_predictions = [{'tarih': date.strftime('%Y-%m-%d'), 'tahmin': round(price, 2)} for date, price in
                             zip(future_dates, future_predictions)]

    return formatted_predictions
