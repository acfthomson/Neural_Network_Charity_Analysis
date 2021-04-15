# Neural_Network_Charity_Analysis

## Overview
Fictional company Alphabet Soup requested a binary classifier that is capable of predicting whether organizations will be successful if Alphabet Soup provides them funding.  Alphabet Soup provided a dataset that contained more than 34,000 organizations that received funding from them.  This dataset captured the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

A neural network model was created to determine if an organization would be successful or not if Alphabet Soup provided them funding.  Alphabet Soup required that the model's target predictive accuracy was 75%.


## Resources
- [charity_data.csv](https://github.com/acfthomson/Neural_Network_Charity_Analysis/tree/main/Resources)
- [sci-kit learn model_selection Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [sci-kit learn preprocessing StandardScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [sci-kit learn preprocessing OneHotEncoder Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [TensorFlow Documentation Documentation](https://www.tensorflow.org/guide)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)


## Dependencies
- Jupyter Notebook
- Python v3.x
    - Pandas
    - OS
    - Sci-Kit Learn
    - TensorFlow
    - Keras


## Results

### Data Preprocessing
- Target variable: IS_SUCCESSFUL
    - This is also known as the dependent variable or 'y'
- Features variables: APPLICATION_TYPE, 'AFFILIATION, 'CLASSIFICATION', USE_CASE, ORGANIZATION, INCOME_AMT, SPECIAL_CONSIDERATIONS
    - Features are also known as independent variables, or 'X'
- Removed variables: EIN and NAME
    -  These were identification columns that did not add value to the model


### Compiling, Training, and Evaluating the Model
#### Initial Model
The initial attempt at achieving a predictive accuracy of 75% used two hidden layers that featured 80 neurons in the first layer and 30 neurons in the second layer.  Both hidden layers utilized the rectified linear unit (ReLU) activation function.  The output layer utilized the sigmoid function.  This gave a total of 6,061 trainable parameters


![initial_model_layers](https://user-images.githubusercontent.com/73897240/114923918-63b39e00-9dfb-11eb-8b07-5c1db7ce24e0.PNG)


This model gave an accuracy score of 72.5%, which did not meet Alphabet Soup's threshold.

![initial_model_score](https://user-images.githubusercontent.com/73897240/114924533-17b52900-9dfc-11eb-95d2-ef11990b4eaa.PNG)


#### First Optimization Attempt


