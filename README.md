# ML-Cancer-Prediction
ML algorithms to predict survivability of patients

## Brief
A researcher wants to predict the survival of cancer patient based on the gene expression data of patients. Develop a model that takes the expression matrix as an input and predict the survival of the patient.

## Data pre-processing

The data that I was given was a txt file containing the information of 57 cancer patients.
For each patient the information known was the weight, age(both in days and years), vital status, days to birth and year of birth, days to death and year of death, clinical stage and a matrix of genes (1 by 60,660). In each column was located the information of each patient, meanwhile the fields were determined on each row.
In order to make the data accessible to the models, it was needed to be normalised.
First, the txt file was read using “tab” as a delimiter to ensure the correct read of the information, and stored as a data set. A row of numbers from 0 to 56 was added as a header to aid in the next step, which would be transposing the table.<br><br>
![image](https://user-images.githubusercontent.com/72141834/172514968-87c0ecc5-2ce8-40d6-a6a2-e9d270dfc034.png)
<br><br>
When accessing information from the data set, python (more specifically, the library “pandas”) prefers each row to be the different patients (cases) and each column to contain the different fields, therefore, a transposition is needed. It is also needed to allow “pandas” to read the correct data type that might otherwise result in the training being less efficient and accurate.<br><br>
 ![image](https://user-images.githubusercontent.com/72141834/172512574-3dafc6a7-c997-44e8-b69d-9d31ea3924b5.png)<br><br>
Once transposed, the data set is saved temporarily onto a file “fixed_table” and then loaded again, allowing python to detect the correct datatype without needing to update manually every row. 
<br><br>
Data types before correct data load:<br>
![image](https://user-images.githubusercontent.com/72141834/172512674-42dd2375-766c-4964-9a54-b13c500d3ff9.png)<br><br>
Data types after correct data load:<br>
![image](https://user-images.githubusercontent.com/72141834/172512686-d4316b3c-0eb8-479f-8645-85db6c6deaab.png)<br><br>


Before being able to split the data for the training and testing sets, the data needs to be encoded as some fields contain other elements that cannot be converted to floats (strings). To overcome this issue, we use the library LabelEncoder from sklearn and encode all of the fields which contain data that can’t be converted to float: demographic_id, vital_status and paper_clinical_stage. We also encode year_of_death and days_to_death as some cells might contain NaN (as the patient might have not died – null values).<br>
The encoding process encodes the data from an object state to an int value (E.g. Alive would be encoded as 0 and Dead would be encoded as 1). Because the majority of the fields were not object fields, no further encoding was needed as the efficiency of the dataset would be good enough to not need further simplification for the models.<br><br>
 ![image](https://user-images.githubusercontent.com/72141834/172512814-abc2eb73-8396-4625-943b-0a52c38b069f.png)
<br><br>
Successively, the data is split into 2 categories, X and y, which are respectively input and output data sets. X contains all of the fields except for vital_status, year_of_death and days_to_death, meanwhile, y contains only the vital_status.
X does not contain year_of_death and days_to_death as the models can use them to predict if a patient died or not. Any data in this field different than NA would result in the patient being dead, making the machine know the result with 100% accuracy, skewing the fairness of the training , therefore they were dropped from the X dataset. <br>
We further split these sets into 2 categories, train sets and test sets. We choose a 20% test set sample size as it allows the models to have enough data to train on, while having enough data to test their accuracy.<br><br>
 ![image](https://user-images.githubusercontent.com/72141834/172512834-987d5c6c-3c5e-45f4-b60d-51a53f7d91cc.png)
<br><br>
X sets are scaled in order to standardise the data, to do so we use the sklearn function StandardScaler.<br>
![image](https://user-images.githubusercontent.com/72141834/172512868-eca74bc0-fb1f-4b8e-8eb7-48293da3fcc2.png)<br>
<br>
Once this has been done, we can finally start to train the models.

## Model selection
A total of 7 models was chosen to be trained and tested on this data set.<br><br>
 ![image](https://user-images.githubusercontent.com/72141834/172512887-c68efda8-6b03-4e80-9b1a-27f5b7c2db24.png)
<br><br>
They are use supervised learning algorithms as the models can be taught by example. Having an input data set and an output data set (X and y) the model has to figure out how to reach the output value from the input data, thus, learning by example (example being the output set).<br>
The algorithms that I chose to apply are:<br>
### •	Decision Trees (Supervised Learning – Classification/Regression)<br>
A decision tree is a flow-chart-like tree structure that uses a branching method to illustrate every possible outcome of a decision. Each node within the tree represents a test on a specific variable – and each branch is the outcome of that test.<br>
### •	Random Forests (Supervised Learning – Classification/Regression)<br>
Random forests or ‘random decision forests’ is an ensemble learning method, combining multiple algorithms to generate better results for classification, regression and other tasks. Each individual classifier is weak, but when combined with others, can produce excellent results. The algorithm starts with a ‘decision tree’ (a tree-like graph or model of decisions) and an input is entered at the top. It then travels down the tree, with data being segmented into smaller and smaller sets, based on specific variables.<br>
### •	Support Vector Machine Algorithm (Supervised Learning - Classification)<br>
Support Vector Machine algorithms are supervised learning models that analyse data used for classification and regression analysis. They essentially filter data into categories, which is achieved by providing a set of training examples, each set marked as belonging to one or the other of the two categories. The algorithm then works to build a model that assigns new values to one category or the other.<br>
### •	Logistic Regression (Supervised learning – Classification)<br>
Logistic regression focuses on estimating the probability of an event occurring based on the previous data provided. It is used to cover a binary dependent variable, that is where only two values, 0 and 1, represent outcomes.<br>
### •	Naïve Bayes Classifier Algorithm (Supervised Learning - Classification)<br>
The Naïve Bayes classifier is based on Bayes’ theorem and classifies every value as independent of any other value. It allows us to predict a class/category, based on a given set of features, using probability.<br>
### •	Nearest Neighbours (Supervised Learning)<br>
The K-Nearest-Neighbour algorithm estimates how likely a data point is to be a member of one group or another. It essentially looks at the data points around a single data point to determine what group it is actually in. For example, if one point is on a grid and the algorithm is trying to determine what group that data point is in (Group A or Group B, for example) it would look at the data points near it to see what group the majority of the points are in.


## Feature engineering
From the dataset it was possible to compare certain fields to help better understand the possible patterns. As an example, here we compare the clinical stage of each patient with the records of them being alive or dead.<br><br>
 ![image](https://user-images.githubusercontent.com/72141834/172512923-0db1974b-fea1-42e2-b010-9a7cb21106f3.png)<br><br>
We can see that the highest probability of survival is located around stage 1 and 2, with a 55% chance of survival at stage 1 and 60% at stage 2. The highest probability of death can be found at stage 3 with a 20% chance of survival, which is 10% lower than the 30% at stage 4.<br>
From the data we can deduce that the patient has the highest chance of survival at stage 2 and the lowest at stage 3. It is worth pointing out that the sample sizes are not even, making the results possibly bias.<br>
We can also plot other fields to grasp the distribution better, and know what to expect when testing the models. Here we have the age distribution over vital status of the patients<br><br>

 ![image](https://user-images.githubusercontent.com/72141834/172512939-a1b88bfd-21f7-4ea7-ab80-9c087dac725f.png)
<br><br>
We can see that the ages range from 51 up to 90. The highest mortality rate is recorded on patients around the ages of 80 to 90, meanwhile the highest survivability is located around the age of 59 to 69. It is safe to assume that a patient x would have the highest probability of survival if he is 59 years old and is at a clinical stage 2, and a patient y would have the highest probability of death if they were either 75, 80 or 90 years old at a clinical stage 3. We can apply the same process on other fields to deduce similar patterns to use when testing the models.<br>

## Parameter tuning
For all the models, a random state of 0 was selected.<br>
In the SVC models, 2 types of kernels were tested: linear and RBF (Radial Basis Function). The linear kernel proved to have a slightly edge over the RBF, having a .4% higher score.<br>
For the tree models, the criterion used was entropy, and for the Random Forest Classifier, the number of estimators was set to 10. That number proved to provide the highest score out of the tests that were carried out for RFC.<br>
In the K-Neighbours model, the number of neighbours was set to 5 in the minkowski metric.<br>
The dataset was split into 20% testing and 80% training. After testing multiple ratios of splits, this proved to yield the highest model score.<br>

## Results
After creating the models, we decide to show each one’s score. The majority of the models scored 100%, with SVC (RBF) scoring 96% and K-Neighbours scoring 80%.<br>
The fact that the majority of models scored 100% is not great as it might prove that there is some overfitting in the dataset.<br>
Another reason why so many algorithms achieved a perfect score could be due to the data sample size.<br><br>
 ![image](https://user-images.githubusercontent.com/72141834/172512952-83849b19-831d-458b-917b-79e6a2b028bc.png)
<br><br>

After displaying the score of each model, they are tested using a confusion matrix, which is a process of evaluating the performance of classification models by comparing the actual targets with those predicted.<br>
Using binary classification, we end up with a 2x2 matrix with 4 values: True Positive, False Positive, False Negative, True Negative.<br><br>
![image](https://user-images.githubusercontent.com/72141834/172513008-e809fc38-d24d-495f-a35c-91e3c009fc3c.png)
<br><br>
Using these values, we can calculate the prediction accuracy of our models by applying this formula:<br>
![image](https://user-images.githubusercontent.com/72141834/172513061-2d61a6e5-61bd-4248-aade-69fd66076434.png)
<br><br>
Our models proved to have a broader range of accuracy, ranging from the low 33% scored by SVC(linear), Logistic Regression and K-Neighbours, to the highest 58% which was achieved by the Random Forest Classfier. The Decision Tree Classifier ranked 2nd with an accuracy of 50%, which is 8% away from the RFC model. In the middle, we find the remaining models SVC(rbf) and GaussianNB with an accuracy of 42%.<br>
![image](https://user-images.githubusercontent.com/72141834/172513085-dcca428b-31df-4d61-a8d3-08843c93528d.png)<br><br>

## Conclusion
The model that proved to be the most accurate out of all the other algorithms was the Random Forest Classifier, with a score of 100% and a prediction accuracy of 58%.<br>
These numbers are not the best as the score of 100% is highly unlikely to be able to apply to a broader range of patients, as some overfitting might be occurring in the dataset.<br>
The overall testing accuracy of all the models was pretty low (peaking at a max of 58%), which indicates that the models might need more training.<br>
In retrospect, the main issue with the model training was with high probability the dataset size. Having only 57 patients and 60000+ fields, the models would struggle to train accurately on a broad range of gene matrix. Perhaps, the best approach to solve this problem would be to apply weights into the training of ANN(artificial neural networks), or simply increase the number of patient records.<br>

## Update - Days left prediction
It is possible now to also predict the days that a patient has left until death if they are correctly classified as dead.<br>
![image](https://user-images.githubusercontent.com/72141834/173203869-ae14a192-7565-4231-a5c3-3b48d5042584.png)

<br><br>
#### Let me know what you think about the project!
