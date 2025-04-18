# Weather Prediction

Weather forecasting is a critical component of many industries, including agriculture, transportation, energy, and event planning. Accurate forecasts help organizations make informed decisions, reduce costs, and ensure safety.

## System Data Flow Diagram (made with eraser.io)

![og](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/image.png)

## Components

- [ML Model for Weather Forecasting](Base_Model_Training.ipynb)

  Random Forest Regression model is used in this problem.

  Output : Temperature

- [Scheduler for Regular Updates](Daily_Model_Trainer.py)

  It updates the data and model from local machine to github repo. by taking data from weather api.

  Using Task Scheduler, Daily_Model_Trainer.py and daily_commit.bat files are run to update the model and commit it to repo. 

- Supaboard Dashboard

  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121400.png)
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121458.png)
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121543.png)
  ![Dashboard_Screenshot](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Supaboard%20Dashboard/Screenshot%202025-04-12%20121633.png)

## ML Model Demo

- Clone repo

```git clone https://github.com/CodeRulerNo1/Weather-Prediction```
- Install essential libraries

```pip install -r requirements.txt```

- Run app.py

```py -m streamlit run app.py```

- Open local website
- Give values to the input parameters
- Press predict button for results

## Visualization

- Data Visualiztion
  
![rain](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Model_img/Screenshot%202025-04-12%20145827.png)
![wea](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Model_img/Screenshot%202025-04-12%20145745.png)
![cov](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Model_img/Screenshot%202025-04-12%20145711.png)

- Model Structure
  
![forest](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Model_img/Screenshot%202025-04-12%20145433.png)

- Model Performance
  
![modelperf](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Model_img/Screenshot%202025-04-12%20145931.png)

- Feature Importance
  
![imp](https://github.com/CodeRulerNo1/Weather-Prediction/blob/main/Model_img/Screenshot%202025-04-12%20145958.png)


## Acknowledgements

- [Dataset Provider](https://www.ncei.noaa.gov/cdo-web/)
- [Weather Api](https://www.weatherapi.com)
- [StreamLit for Model Demo](https://streamlit.io)

## Author

- [@Ichhabal Singh](https://www.github.com/CodeRulerNo1)
