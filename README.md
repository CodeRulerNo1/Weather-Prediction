# Weather Prediction

Weather forecasting is a critical component of many industries, including agriculture, transportation, energy, and event planning. Accurate forecasts help organizations make informed decisions, reduce costs, and ensure safety.

## Components

- [ML Model for Weather Forecasting](Base_Model_Training.ipynb)
  
  Random Forest Regression model is used in this problem.
  Output : Temperature
  
- [Scheduler for Regular Updates](Daily_Model_Trainer.py)
  
  It updates the data and model from local machine to github repo.
  
- Supaboard Dashboard
  
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121400.png)
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121458.png)
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121543.png)
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121633.png)

## ML Model Demo
- Install essential libraries
pip install -r requirements.txt
- Clone repo
git clone https://github.com/CodeRulerNo1/Weather-Prediction
- Run app.py
py -m streamlit run app.py
- Open local website
- Give values to the input parameters
- Press predict button for results

## Acknowledgements

- [Dataset Provider](https://www.ncei.noaa.gov/cdo-web/)
- [Weather Api](https://www.weatherapi.com)
- [StreamLit for Model Demo](https://streamlit.io)

## Authors

- [@Ichhabal Singh](https://www.github.com/CodeRulerNo1)
