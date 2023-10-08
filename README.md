# Restaurant Recommendation Dialogue System
This program provides the user with a restaurant recommendation based on the preferences the user provides.
The program will run with the command "python main.py" in the terminal.

## Required packages
The program uses functions from the following packages:
- sklearn
- pickle
- numpy
- tensorflow
- matplotlib
- tranformers
- enum
- levenshtein
- pandas
- os
- time

## Configurations
The program has four configuration options for the dialogue system.
- All system output in capital letters.
- Add a delay to the  system output.
- Using the Levenshtein distance to account for spelling errors.
- Checking whether the result of the Levenshtein matching is correct with  the meaning of the user.
These configurations can be set in dialogue_system.py in the init function of the dialogue system class. These contain four booleans, one for each configuration. Switching the values of the booleans will turn the options on and off.

## More detailed description of the project
The aim of this project is creating a goal-oriented dialog agent that is used for restaurant recommendations. The user of the system can provide preferences for the type of food, price range, area, food quality, length of stay and crowdedness. Then the system provides the user with a list of suitable restaurants. This is set up as a dialog system, the system asks for preferences and the user answers its questions to get a fitting recommendation.
The system classifies user utterances to fit  categories, e.g. greeting, informing and affirming. These utterances can be classified using multiple approaches. In this project two baseline methods and three machine learning classifiers are used. These machine learning classifiers are a decision tree, logistic regression and a feed-forward neural network. All three machine learning models have a high accuracy, around 0.98, but there are some differences in other performance metrics.
The dialog system is implemented according to a dialog state transition diagram. The system will inquire about the type of cuisine, price and location that is preferred and give the available options. Additionally, the system can also provide extra information like restaurant address and telephone number.
Furthermore, a reasoning component is integrated in the system allowing for inferences based on existing restaurant properties to determine new ones. Finally, four options that allow for configurablity have been added for a better experience for the user.