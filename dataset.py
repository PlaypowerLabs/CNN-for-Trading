import datetime
import os

import cv2
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from tqdm import tqdm

import utils as _U

SUPPORTED_INDICATORS = ["MA"]
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)


def cal_indicators(tabular_df, indicator_name, parameter):
    if indicator_name == "MA":
        slice_win_size = int(parameter)
        MA = tabular_df["Close"].rolling(slice_win_size, min_periods=1).mean()
        return MA  # pd.Series

def single_symbol_latest_image(
    tabular_df, image_size, indicators, show_volume, lookback_days=60
):
    """Generate a single image using the latest lookback_days of data for a symbol
    
    parameters: [
        tabular_df     -> pandas.DataFrame: tabular data for a single symbol,
        image_size     -> tuple: (H, W), should be (128, 180) for 60-day lookback,
        indicators     -> dict: technical indicators added on the image, e.g. {"MA": [20]},
        show_volume    -> boolean: show Volume bars or not,
        lookback_days  -> int: number of days to look back (default 60)
    ]
    
    return -> [np.array(image_size), symbol, latest_date] or None if insufficient data
    """
    
    # Get the latest lookback_days of data
    if len(tabular_df) < lookback_days:
        print(f"Warning: Not enough data for symbol {tabular_df.iloc[0]['Symbol']}. Available: {len(tabular_df)}, Required: {lookback_days}")
        return None
    
    # Take the latest lookback_days
    latest_data = tabular_df.tail(lookback_days).copy()
    
    ind_names = []
    if indicators:
        for ind_dict in indicators:
            for ind, params in ind_dict.items():
                assert ind in SUPPORTED_INDICATORS, f"Error: {ind} not supported"
                ind_names.append(ind)
                latest_data[ind] = cal_indicators(
                    latest_data,
                    ind,
                    params if not isinstance(params, list) else params,
                )

    # Check for no transaction days
    if (
        1.0
        * (
            latest_data[["Open", "High", "Low", "Close"]].sum(axis=1)
            / latest_data["Open"]
            == 4
        )
    ).sum() > lookback_days // 5:
        print(f"Warning: Too many no-transaction days for symbol {latest_data.iloc[0]['Symbol']}")
        return None

    # Project price into quantile
    price_slice = latest_data[["Open", "High", "Low", "Close"] + ind_names].reset_index(drop=True)
    volume_slice = latest_data[["Volume"]].reset_index(drop=True)
    
    price_slice = (price_slice - np.min(price_slice.values)) / (
        np.max(price_slice.values) - np.min(price_slice.values)
    )
    volume_slice = (volume_slice - np.min(volume_slice.values)) / (
        np.max(volume_slice.values) - np.min(volume_slice.values)
    )

    if not show_volume:
        price_slice = price_slice.apply(lambda x: x * (image_size[0] - 1)).astype(int)
    else:
        if image_size[0] == 32:
            price_slice = price_slice.apply(lambda x: x * (25 - 1) + 7).astype(int)
            volume_slice = volume_slice.apply(lambda x: x * (6 - 1)).astype(int)
        else:
            price_slice = price_slice.apply(lambda x: x * (51 - 1) + 13).astype(int)
            volume_slice = volume_slice.apply(lambda x: x * (12 - 1)).astype(int)

    # Create the image
    image = np.zeros(image_size)
    for i in range(len(price_slice)):
        # draw candlestick
        image[price_slice.loc[i]["Open"], i * 3] = 255.0
        image[
            price_slice.loc[i]["Low"] : price_slice.loc[i]["High"] + 1, i * 3 + 1
        ] = 255.0
        image[price_slice.loc[i]["Close"], i * 3 + 2] = 255.0
        # draw indicators
        for ind in ind_names:
            image[price_slice.loc[i][ind], i * 3 : i * 3 + 2] = 255.0
        # draw Volume bars
        if show_volume:
            image[: volume_slice.loc[i]["Volume"], i * 3 + 1] = 255.0

    symbol = latest_data.iloc[-1]["Symbol"]
    latest_date = latest_data.iloc[-1]["Date"]
    
    return [image, symbol, latest_date]


def single_symbol_image(
    tabular_df, image_size, start_date, sample_rate, indicators, show_volume, mode
):
    """generate Candlelist images

    parameters: [
        tabular_df  -> pandas.DataFrame: tabular data,
        image_size  -> tuple: (H, W), size shouble (32, 15), (64, 60)
        start_date  -> int: truncate extra rows after generating images,
        indicators  -> dict: technical indicators added on the image, e.g. {"MA": [20]},
        show_volume -> boolean: show Volume bars or not
        mode        -> 'train': for train & validation; 'test': for test; 'inference': for inference
    ]

    Note: A single day's data occupies 3 pixel (width). First rows's dates should be prior to the start Date in order to make sure there are enough data to generate image for the start Date.

    return -> list: each item of the list is [np.array(image_size), binary, binary, binary, binary]. The last three binary (0./1.) are the labels of ret5, ret20, ret60

    """

    ind_names = []
    if indicators:
        for ind_dict in indicators:
            for ind, params in ind_dict.items():
                assert ind in SUPPORTED_INDICATORS, f"Error: {ind} not supported"
                ind_names.append(ind)
                tabular_df[ind] = cal_indicators(
                    tabular_df,
                    ind,
                    params if not isinstance(params, list) else params,
                )

    dataset = []
    valid_dates = []
    lookback = image_size[1] // 3
    for d in range(lookback - 1, len(tabular_df)):
        # random skip some trading dates
        if np.random.rand(1) > sample_rate:
            continue
        # skip dates before start_date
        if tabular_df.iloc[d]["Date"] < start_date:
            continue

        price_slice = tabular_df[d - (lookback - 1) : d + 1][
            ["Open", "High", "Low", "Close"] + ind_names
        ].reset_index(drop=True)
        volume_slice = tabular_df[d - (lookback - 1) : d + 1][["Volume"]].reset_index(
            drop=True
        )

        # number of no transactions days > 0.2*look back days
        if (
            1.0
            * (
                price_slice[["Open", "High", "Low", "Close"]].sum(axis=1)
                / price_slice["Open"]
                == 4
            )
        ).sum() > lookback // 5:
            continue

        valid_dates.append(
            tabular_df.iloc[d]["Date"]
        )  # trading dates surviving the validation

        # project price into quantile
        price_slice = (price_slice - np.min(price_slice.values)) / (
            np.max(price_slice.values) - np.min(price_slice.values)
        )
        volume_slice = (volume_slice - np.min(volume_slice.values)) / (
            np.max(volume_slice.values) - np.min(volume_slice.values)
        )

        if not show_volume:
            price_slice = price_slice.apply(lambda x: x * (image_size[0] - 1)).astype(
                int
            )
        else:
            if image_size[0] == 32:
                price_slice = price_slice.apply(lambda x: x * (25 - 1) + 7).astype(int)
                volume_slice = volume_slice.apply(lambda x: x * (6 - 1)).astype(int)
            else:
                price_slice = price_slice.apply(lambda x: x * (51 - 1) + 13).astype(int)
                volume_slice = volume_slice.apply(lambda x: x * (12 - 1)).astype(int)

        image = np.zeros(image_size)
        for i in range(len(price_slice)):
            # draw candlelist
            image[price_slice.loc[i]["Open"], i * 3] = 255.0
            image[
                price_slice.loc[i]["Low"] : price_slice.loc[i]["High"] + 1, i * 3 + 1
            ] = 255.0
            image[price_slice.loc[i]["Close"], i * 3 + 2] = 255.0
            # draw indicators
            for ind in ind_names:
                image[price_slice.loc[i][ind], i * 3 : i * 3 + 2] = 255.0
            # draw Volume bars
            if show_volume:
                image[: volume_slice.loc[i]["Volume"], i * 3 + 1] = 255.0

        label_ret5 = 1 if np.sign(tabular_df.iloc[d]["ret5"]) > 0 else 0
        label_ret20 = 1 if np.sign(tabular_df.iloc[d]["ret20"]) > 0 else 0
        label_ret60 = 1 if np.sign(tabular_df.iloc[d]["ret60"]) > 0 else 0

        entry = [
            image,
            label_ret5,
            label_ret20,
            label_ret60,
            tabular_df.iloc[d]["Date"],
            tabular_df.iloc[d]["Symbol"],
        ]
        dataset.append(entry)

    if mode == "train" or mode == "test":
        return dataset
    else:
        return [tabular_df.iloc[0]["Symbol"], dataset, valid_dates]


class ImageDataSet:
    def __init__(
        self,
        win_size,
        start_date,
        end_date,
        mode,
        label,
        indicators=[],
        show_volume=False,
        parallel_num=-1,
    ):
        ## Check whether inputs are valid
        assert isinstance(start_date, int) and isinstance(
            end_date, int
        ), f"Type Error: start_date & end_date shoule be int"
        assert (
            start_date < end_date
        ), f"start Date {start_date} cannnot be later than end Date {end_date}"
        assert win_size in [5, 20, 60], f"Wrong look back days: {win_size}"
        assert mode in ["train", "test", "inference"], f"Type Error: {mode}"
        assert label in ["RET5", "RET20", "RET60"], f"Wrong Label: {label}"
        assert indicators is None or isinstance(
            indicators, list
        ), "Config Error: indicators should be a dict like [{'MA': 20}]"

        ## Attributes of ImageDataSet
        if win_size == 5:
            self.image_size = (32, 15)
            self.extra_dates = datetime.timedelta(days=40)
        elif win_size == 20:
            self.image_size = (64, 60)
            self.extra_dates = datetime.timedelta(days=40)
        else:  # win_size == 60
            self.image_size = (128, 180)
            self.extra_dates = datetime.timedelta(days=80)

        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.label = label
        self.indicators = indicators
        self.show_volume = show_volume
        self.parallel_num = parallel_num

        ## Load data from zipfile
        self.load_data()

        # Log info
        if indicators:
            ind_info = []
            for ind_dict in indicators:
                for ind, params in ind_dict.items():
                    if isinstance(params, list):
                        ind_info.append((ind, [str(p) for p in params]))
                    else:
                        ind_info.append((ind, [str(params)]))
        else:
            ind_info = []
        print(
            f"DataSet Initialized\n \t - Mode:         {self.mode.upper()}\n \t - Image Size:   {self.image_size}\n \t - Time Period:  {self.start_date} - {self.end_date}\n \t - Indicators:   {ind_info}\n \t - Volume Shown: {self.show_volume}"
        )

    @_U.timer("Load Data", "8")
    def load_data(self):
        path = "data/"
        print("Path to dataset files:", path)

        # Find CSV file
        csv_file = None
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_file = os.path.join(path, file)
                break
        if not csv_file:
            print("No CSV file found in the downloaded dataset.")
            return None

        # Load CSV
        tabularDf = pd.read_csv(csv_file)
        print(f"Loaded dataset with shape: {tabularDf.shape}")

        # Parse 'Date' column to datetime
        tabularDf["Date"] = pd.to_datetime(tabularDf["Date"], format="%Y-%m-%d")

        # Padding for extra dates
        padding_start_date = pd.to_datetime(str(self.start_date)) - self.extra_dates
        padding_end_date = pd.to_datetime(str(self.end_date)) + self.extra_dates

        self.df = tabularDf.loc[
            (tabularDf["Date"] > padding_start_date)
            & (tabularDf["Date"] < padding_end_date)
        ].copy(deep=False)
        tabularDf = []  # clear memory

        self.df["ret5"] = np.zeros(self.df.shape[0])
        self.df["ret20"] = np.zeros(self.df.shape[0])
        self.df["ret60"] = np.zeros(self.df.shape[0])
        self.df["ret5"] = (self.df["Close"].pct_change(5) * 100).shift(-5)
        self.df["ret20"] = (self.df["Close"].pct_change(20) * 100).shift(-20)
        self.df["ret60"] = (self.df["Close"].pct_change(60) * 100).shift(-60)
        self.start_date = pd.to_datetime(str(self.start_date))
        self.end_date = pd.to_datetime(str(self.end_date))
        self.df = self.df.loc[self.df["Date"] <= self.end_date]

    def generate_images(self, sample_rate):
        dataset_all = Parallel(n_jobs=self.parallel_num)(
            delayed(single_symbol_image)(
                g[1],
                image_size=self.image_size,
                start_date=self.start_date,
                sample_rate=sample_rate,
                indicators=self.indicators,
                show_volume=self.show_volume,
                mode=self.mode,
            )
            for g in tqdm(
                self.df.groupby("Symbol"),
                desc=f"Generating Images (sample rate: {sample_rate})",
            )
        )

        if self.mode == "train" or self.mode == "test":
            image_set = []
            for symbol_data in dataset_all:
                image_set = image_set + symbol_data
            dataset_all = []  # clear memory

            if self.mode == "train":  # resample to handle imbalance
                image_set = pd.DataFrame(
                    image_set, columns=["img", "ret5", "ret20", "ret60", "Date", "Symbol"]
                )
                image_set["index"] = image_set.index
                smote = SMOTE()
                if self.label == "RET5":
                    num0_before = image_set.loc[image_set["ret5"] == 0].shape[0]
                    num1_before = image_set.loc[image_set["ret5"] == 1].shape[0]
                    resample_index, _ = smote.fit_resample(
                        image_set[["index", "ret20", "ret60"]], image_set["ret5"]
                    )
                    image_set = image_set[["img", "ret5", "ret20", "ret60", "Date", "Symbol"]].loc[
                        resample_index["index"]
                    ]
                    num0 = image_set.loc[image_set["ret5"] == 0].shape[0]
                    num1 = image_set.loc[image_set["ret5"] == 1].shape[0]
                    image_set = image_set.values.tolist()

                elif self.label == "RET20":
                    num0_before = image_set.loc[image_set["ret20"] == 0].shape[0]
                    num1_before = image_set.loc[image_set["ret20"] == 1].shape[0]
                    resample_index, _ = smote.fit_resample(
                        image_set[["index", "ret5", "ret60"]], image_set["ret20"]
                    )
                    image_set = image_set[["img", "ret5", "ret20", "ret60", "Date", "Symbol"]].loc[
                        resample_index["index"]
                    ]
                    num0 = image_set.loc[image_set["ret20"] == 0].shape[0]
                    num1 = image_set.loc[image_set["ret20"] == 1].shape[0]
                    image_set = image_set.values.tolist()

                else:  # self.label == "RET60"
                    num0_before = image_set.loc[image_set["ret60"] == 0].shape[0]
                    num1_before = image_set.loc[image_set["ret60"] == 1].shape[0]
                    resample_index, _ = smote.fit_resample(
                        image_set[["index", "ret5", "ret20"]], image_set["ret60"]
                    )
                    image_set = image_set[["img", "ret5", "ret20", "ret60", "Date", "Symbol"]].loc[
                        resample_index["index"]
                    ]
                    num0 = image_set.loc[image_set["ret60"] == 0].shape[0]
                    num1 = image_set.loc[image_set["ret60"] == 1].shape[0]
                    image_set = image_set.values.tolist()

                print(
                    f"LABEL: {self.label}\n\tBefore Resample: 0: {num0_before}/{num0_before+num1_before}, 1: {num1_before}/{num0_before+num1_before}\n\tResampled ImageSet: 0: {num0}/{num0+num1}, 1: {num1}/{num0+num1}"
                )

            return image_set

        else:
            return dataset_all


# reload(_U)  # Remove or replace if not needed
def main():
    dataset = ImageDataSet(
        win_size=60,  # Lookback window (5 or 20)
        start_date=20050101,  # Example start date (YYYYMMDD int)
        end_date=20050105,  # Example end date (YYYYMMDD int)
        mode="train",  # "train", "test", or "inference"
        label="RET5",  # "RET5" or "RET20"
        indicators=[{"MA": 20}],  # e.g. [{"MA": 20}] if needed
        show_volume=True,
        parallel_num=-1,  # Use all CPUs
    )
    images = dataset.generate_images(sample_rate=1.0)
    print(f"Generated {len(images)} images using ImageDataSet class.")
    for i, (img, ret5, ret20, date, code) in enumerate(images):
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime("%Y%m%d")
        else:
            date_str = str(date)
        filename = (
            f"{output_dir}/{code}_{date_str}_img_{i}_ret5_{ret5}_ret20_{ret20}.png"
        )
        success = cv2.imwrite(filename, img)
        if not success:
            print(f"Failed to save {filename}, shape={img.shape}, dtype={img.dtype}")


if __name__ == "__main__":
    main()
