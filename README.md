# Deep Transformer Networks. 
Applying a transformer-based model to classify the levels of Ozone in center of Madrid (Plaza España station). We also make the predictions with other machine learning models such as Multi Layer Perceptron Netwroks, Long-short term memory networks or Random Forest. 

## Data
Data was extracted from public sources (http://www.aemet.es/ and https://datos.madrid.es/portal/site/egob). 
Data was previously preprocessed. Previous values were imputed in NA and in outlier cases (3Q + 3*IQR and 1Q - 3*IQR).
The predictor variables used are:

-Hourly NO concentration in Plaza España Station. 

-Hourly NO2 concentration in Plaza España Station. 

-Hourly NOx concentration in Plaza España Station. 

-Hourly CO concentration in Plaza España Station. 

-Hourly SO2 concentration in Plaza España Station. 

-Hourly O3 concentration in Plaza España Station. 

-Daily rain in El Retiro Station (the closest meteorological station to Plaza España station). 

-Daily maximum temperature in El Retiro Station. 

-Daily minimum temperature in El Retiro Station. 

-Daily average temperature in El Retiro Station. 

-Daily maximum pressure El Retiro Station. 

-Daily minimum pressure El Retiro Station. 

-Weekday.

-Type of day (diary day, saturday, sunday, public holyday). 
