from __init__ import *
import utils as _U
reload(_U)
import model as _M
reload(_M)
import dataset as _D
reload(_D)


def model_inference(model, setting):
    
    model.eval()
    
    # Load your data
    data_path = '/'
    
    # Find CSV file
    csv_file = None
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            csv_file = os.path.join(data_path, file)
            break
    if not csv_file:
        print("No CSV file found in the dataset path.")
        return None

    # Load CSV
    tabularDf = pd.read_csv(csv_file)
    print(f"Loaded dataset with shape: {tabularDf.shape}")

    # Parse 'Date' column to datetime
    tabularDf["Date"] = pd.to_datetime(tabularDf["Date"], format="%d-%m-%Y")
    
    # Use the latest date as end_date and go back 85 days for start_date
    latest_date = tabularDf['Date'].max()
    start_date = latest_date - datetime.timedelta(days=85) ## 85 to have enough data for 60-day lookback and some buffer
    
    # Filter data for the last 60 days
    df = tabularDf.loc[
        (tabularDf["Date"] >= start_date) & (tabularDf["Date"] <= latest_date)
    ].copy(deep=False)
    
    print(f"Using data from {start_date.date()} to {latest_date.date()}")
    print(f"Filtered dataset shape: {df.shape}")
    
    # Generate single images per symbol
    image_size = (128, 180)  # For 60-day lookback
    lookback_days = 60
    
    symbol_factors = pd.DataFrame([], columns=['Symbol', 'Date', 'up_factor'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for symbol, symbol_data in tqdm(df.groupby("Symbol"), desc="Generating single images per symbol"):
        if len(symbol_data) < lookback_days:
            print(f"Skipping symbol {symbol}: insufficient data ({len(symbol_data)} < {lookback_days})")
            continue
            
        # Sort by date to ensure we get the latest data
        symbol_data = symbol_data.sort_values('Date')
        
        result = _D.single_symbol_latest_image(
            symbol_data,
            image_size=image_size,
            indicators=setting.DATASET.INDICATORS,
            show_volume=setting.DATASET.SHOW_VOLUME,
            lookback_days=lookback_days
        )
        
        if result is not None:
            image, symbol, latest_date = result
            
            # Run inference on this single image
            input_tensor = torch.Tensor(np.array([image]))
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)[:, 1]  # Get probability of positive class
                up_factor = output.item()
            
            # Add to results
            new_row = pd.DataFrame({
                'Symbol': [symbol],
                'Date': [latest_date],
                'up_factor': [up_factor]
            })
            symbol_factors = pd.concat([symbol_factors, new_row], axis=0)
    
    print(f"Generated predictions for {len(symbol_factors)} symbols")
    return symbol_factors


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train Models via YAML files')
    parser.add_argument('setting', type=str, \
                        help='Experiment Settings, should be yaml files like those in /configs')

    args = parser.parse_args()

    with open(args.setting, 'r') as f:
        setting = _U.Dict2ObjParser(yaml.safe_load(f)).parse()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert setting.MODEL in ['CNN5d', 'CNN20d', 'CNN60d'], f"Wrong Model Template: {setting.MODEL}"

    if 'factors' not in os.listdir('./'):
        os.system('mkdir factors')
    if setting.INFERENCE.FACTORS_SAVE_FILE.split('/')[1] not in os.listdir('./factors/'):
        os.system(f"cd factors && mkdir {setting.INFERENCE.FACTORS_SAVE_FILE.split('/')[1]}")
        
    if setting.MODEL == 'CNN5d':
        model = _M.CNN5d()
    elif setting.MODEL == 'CNN20d':
        model = _M.CNN20d()
    else:
        model = _M.CNN60d()
    model.to(device)

    state_dict = torch.load(setting.TRAIN.MODEL_SAVE_FILE)
    model.load_state_dict(state_dict['model_state_dict'])

    factors = model_inference(model, setting)
    factors.to_csv(setting.INFERENCE.FACTORS_SAVE_FILE)