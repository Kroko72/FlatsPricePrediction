import numpy as np
import requests
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from .const import MOSCOW_METRO_DISTANCES, MOSCOW_DISTRICTS
from math import radians, cos, sin, asin, sqrt  # needed for calculate distance


# Function to calculate distance between two points
def distance(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / -2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class MoscowModel:
    def __init__(self) -> None:
        pd.options.mode.chained_assignment = None
        self.model = RandomForestRegressor(n_estimators=177, min_samples_split=6, min_samples_leaf=16, max_depth=21,
                                           n_jobs=-1)  # parameters from moscow flats price prediction project

        # dataset preparation
        df_base = pd.read_csv('flats.csv', sep=',', encoding="utf-8")
        df = df_base.drop(
            ['author', 'author_type', 'link', 'deal_type', "residential_complex", "city", "accommodation_type",
             "house_number"], axis=1)
        inds = df[["underground", "district"]].isnull().all(axis=1)
        df = df.loc[~inds, :]
        df = df[df['underground'].notna()]
        df["price"] = np.log2(df['price'])
        df = df[df.price > 15.0]

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(df["district"])
        self.disctrict_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        df['district'] = label_encoder.transform(df['district'])
        df["district"].replace({142: np.nan}, inplace=True)
        df["district"].interpolate(inplace=True)
        df["district"].iloc[0] = 52.0
        df["district"] = df["district"].round(0).astype(int)

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(df["street"])
        self.street_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        df['street'] = label_encoder.transform(df['street'])

        df["street"].replace({2476: np.nan}, inplace=True)
        df["street"].interpolate(inplace=True)
        df["street"] = df["street"].round(0).astype(int)
        df["rooms_count"].replace({-1: np.nan}, inplace=True)
        df["rooms_count"].interpolate(inplace=True)
        df["rooms_count"][0] = 2.0
        df["rooms_count"][1] = 1.0

        df['distance_to_centre'] = 0.0  # better to calculate by the address
        for i, j in df.iterrows():
            df["distance_to_centre"][i] = MOSCOW_METRO_DISTANCES[j["underground"]]

        df = df.drop(["level_0", "index"], axis=1)
        print(df.dtypes)
        print("-----------")
        print(df.isna().sum())

        # learning
        X = df.drop(columns=["price", "underground", "price_per_m2"])
        print(X.columns)
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        print('R2:', r2_score(y_test, self.model.predict(X_test)))

    def distance_to_centre(self, address: str) -> float:
        lo, la = map(float, requests.get(f"https://geocode-maps.yandex.ru/1.x/?apikey=e28e8807-9870-4b03-8461-20d60bdd95d3&geocode=Москва {address}&format=json").json()["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["Point"]['pos'].split())
        d = distance(la, 55.755863, lo, 37.617700)
        return d

    def calculate(self, floors_count: int, floor: int, street: str, house: str, district: str,
                  year_of_construction: int, living_metr: float, kitchen_metr: float, total_meters: float,
                  rooms_count: int) -> float:
        # Check that we have this street in dictionary
        try:
            street_number = self.street_name_mapping[street]
        except KeyError:
            street_number = max(self.street_name_mapping.values()) + 1  # is it ok?
        X_flat = [[kitchen_metr, living_metr, year_of_construction, floor, floors_count,
                   rooms_count, total_meters, MOSCOW_DISTRICTS[district.lower()],
                   street_number, self.distance_to_centre(f"{street}, {house}")]]
        return int(2**self.model.predict(X_flat)[0])
