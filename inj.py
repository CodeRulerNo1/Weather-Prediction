
from __future__ import print_function
import joblib
import weatherapi
from weatherapi.rest import ApiException
from pprint import pprint
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

today = datetime.today()
future_date = today + timedelta(days=14)
future_date_str = future_date.strftime('%Y-%m-%d')  # Example format: YYYY-MM-DD
# Configure API key authorization: ApiKeyAuth
configuration = weatherapi.Configuration()
configuration.api_key['key'] = 'd30ec8979f604521ac4144452251104'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['key'] = 'Bearer'

# create an instance of the API class
api_instance = weatherapi.APIsApi(weatherapi.ApiClient(configuration))
q = 'amritsar' # str | Pass US Zipcode, UK Postcode, Canada Postalcode, IP address, Latitude/Longitude (decimal degree) or city name. Visit [request parameter section](https://www.weatherapi.com/docs/#intro-request) to learn more.
days = 1 # int | Number of days of weather forecast. Value ranges from 1 to 14
dt = future_date_str # date | Date should be between today and next 14 day in yyyy-MM-dd format. e.g. '2015-01-01' (optional)
unixdt = 56 # int | Please either pass 'dt' or 'unixdt' and not both in same request. unixdt should be between today and next 14 day in Unix format. e.g. 1490227200 (optional)
hour = 56 # int | Must be in 24 hour. For example 5 pm should be hour=17, 6 am as hour=6 (optional)
lang = 'lang_example' # str | Returns 'condition:text' field in API in the desired language.<br /> Visit [request parameter section](https://www.weatherapi.com/docs/#intro-request) to check 'lang-code'. (optional)
alerts = 'alerts_example' # str | Enable/Disable alerts in forecast API output. Example, alerts=yes or alerts=no. (optional)
aqi = 'aqi_example' # str | Enable/Disable Air Quality data in forecast API output. Example, aqi=yes or aqi=no. (optional)
tp = 56 # int | Get 15 min interval or 24 hour average data for Forecast and History API. Available for Enterprise clients only. E.g:- tp=15 (optional)

try:
    # Forecast API
    api_response = api_instance.forecast_weather(q, days, dt=dt, unixdt=unixdt, hour=hour, lang=lang, alerts=alerts, aqi=aqi, tp=tp)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling APIsApi->forecast_weather: %s\n" % e)
weather_data = api_response


data = {}
data['name'] = [weather_data['location']['name']]
current = weather_data['current']
data['temp'] = [current.get('temp_c')]
data['feelslike'] = [current.get('feelslike_c')]
data['dew'] = [current.get('dewpoint_c')]
data['humidity'] = [current.get('humidity')]
data['precip'] = [current.get('precip_mm')]
data['windgust'] = [current.get('gust_kph')]
data['windspeed'] = [current.get('wind_kph')]
data['winddir'] = [current.get('wind_dir')]
data['sealevelpressure'] = [current.get('pressure_mb')]
data['cloudcover'] = [current.get('cloud')]
data['visibility'] = [current.get('vis_km')]
data['uvindex'] = [current.get('uv')]
data['conditions'] = [current['condition'].get('text')]
data['description'] = [current['condition'].get('text')] 
data['icon'] = [current['condition'].get('icon')]

df = pd.DataFrame(data)
df0 = pd.read_csv(r'D:\project\Hackathon\Punjab 2022-09-01 to 2025-04-26.csv')
df1= pd.read_csv(r'D:\project\Hackathon\weather_data.csv')
df0 = df0.set_index("datetime")

def wind_direction_to_degrees(direction):
    """Converts wind direction from cardinal to degrees."""
    direction_mapping = {
        'N': 0,
        'NNE': 22.5,
        'NE': 45,
        'ENE': 67.5,
        'E': 90,
        'ESE': 112.5,
        'SE': 135,
        'SSE': 157.5,
        'S': 180,
        'SSW': 202.5,
        'SW': 225,
        'WSW': 247.5,
        'W': 270,
        'WNW': 292.5,
        'NW': 315,
        'NNW': 337.5
    }
    return direction_mapping.get(direction, direction)  # Return original if not found


# Assuming your DataFrame is named 'df' and has a column named 'winddirection'
df['winddir'] = df['winddir'].apply(wind_direction_to_degrees)


# Load the RandomForestRegressor model
rf_regressor = joblib.load(r'D:\project\Hackathon\rf_regressor_model (1).pkl')


# Assuming 'features_cols' and 'target_col' are defined as before
features_cols = ['cloudcover','winddir','sealevelpressure','windgust','humidity', 'windspeed', 'precip', 'uvindex','dew']
df=pd.concat([df, df1], axis=0, ignore_index=True)
df.to_csv(r'D:\project\Hackathon\weather_data.csv', index=False)
X = df0[features_cols]
y = df0['temp']
X_new = df0[features_cols]
y_new = df0['temp'] 

combined_x = pd.concat([X, X_new], axis=0, ignore_index=True)
combined_y = pd.concat([y, y_new], axis=0, ignore_index=True)

mask = combined_x.notna().all(axis=1)

# Apply the mask to both X and y
combined_x = combined_x[mask]
combined_y = combined_y[mask]

X_train, X_test, y_train, y_test = train_test_split(combined_x, combined_y, test_size=0.2, random_state=42)

rf_regressor.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



# Save the RandomForestRegressor model
joblib.dump(rf_regressor, r'D:\project\Hackathon\rf_regressor_model (1).pkl')



# Assuming 'y_test' contains the actual values and 'y_pred' contains the predicted values
plt.figure(figsize=(8, 6))
# Scatter plot for original predictions
plt.scatter(y_test, y_pred, alpha=0.7, label='Original Predictions')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()



