# PysparkProj-MultiClass-Classification

## 1. Introduction

- This Project is uptaken as an extension to the [KickstarterProjects-Classification](https://github.com/uma-sid/KickstarterProjects-Classification)
- The previous project is done as a binary classification. But, in this project the classification is performed as a Multiclass classification, by not dropping the 'Suspended' and 'Cancelled' State projects. 
- This project is done using PySpark. Kaggle notebooks are used to complete this project.
- The cleaned dataset obtained from the previous project [KickstarterProjects-Classification](https://github.com/uma-sid/KickstarterProjects-Classification) is used for this project.

## 2. PCA & clustering

- After importing the cleaned dataset, PCA is performed on the dataframe with k=2 as we are interested to take only 2 principal components so that the obtained components can be used for clustering, which is easier for visualization. 

- During the clustering, best number of clusters is verified using the Silhouette score. It is observed that the best silhouette score occurred at 2.

<img width="533" alt="image" src="https://user-images.githubusercontent.com/98278525/193662077-9048142d-1d52-4d69-b2d8-8e900414f734.png">

- K-means clustering algorithm is applied on the obtained two principal components by taking the k value as 2.![image](https://user-
![image](https://user-images.githubusercontent.com/98278525/193662197-89ea0241-51d4-441b-ae74-ce83926c09d3.png)

However, the formed clusters seemed to be inseparable. So, again the Silhouette coefficient is calculated, which turned out to be 0.4. The Silhouette coefficient near to zero indicates that the clusters are inseparable or barely separable. It can be also be represented as the distance between the clusters is insignificant. But the score here is less than 0.5 so, we can conclude that the clusters are identifiable but the distance between them is not significant.

## 3. High Performance Computational Implementation:
Pyspark has been chosen to implement the machine learning predictions, which also serves the high performance computational needs required for a huge amount of data. “Multinomial Logistics Regression” is used to predict the states of the projects, as the dataset is of imbalanced nature with high success and failed states and very few canceled and suspended states of projects, ‘Oversampling’ technique is used to balance the dataset. The complete implementation is explained in the ipynb notebook. 

Data Balance before and after oversampling is shown below.

<img width="511" alt="image" src="https://user-images.githubusercontent.com/98278525/193663558-20a501e1-8028-4d65-879b-de91a70736b6.png">

## 4. Performance evaluation & Conclusion
The trained model is applied on the test set to predict the classes. The Area under ROC curve turned out to be 0.58 which indicates that there is a chance of only 58% that the model will correctly distinguish between the classes. The other metrics are shown below.

<img width="681" alt="image" src="https://user-images.githubusercontent.com/98278525/193663839-5b215b43-b115-432d-8063-80eb0dd0b682.png">

Class-0(Successful Projects): Although the true positive rate lies around 0.92, it is the precision that tells the actual precision in calculating the positives is around 0.84, which is still a high number and acceptable. Also the fact that f-measure lies at 0.88 tells that the model did a good job in predicting the Successful state projects. The model is capable of predicting this class more accurately than all others. 
Class-1(Failed Projects): The true positive rate, precision and the f-measure, all lying around 0.5 indicates that the model isn’t good in classifying the Failed state project. 
Class-2(Suspended Projects): Although the true positive rate is at 0.6, the f-measure and precision indicates the poor performance of the model in predicting the Suspended state projects. 
Class-3(Cancelled Projects): All the metrics for this class are below par which  indicates that the model completely failed to predict the Cancelled projects. 

The accuracy of the overall model lies around 59.65%. But, as the model failed to predict three of the classes at least with an acceptable accuracy, we can make three inferences here. Its either the data is too volatile and it cannot be modelled with good accuracy or the projects are in those states for the reasons which are outside the scope of this dataset or the model isn’t the right fit for this dataset.  
When looked in to the Kickstarter’s ‘About’ section of the platform it was found that the main states of the projects are expected to be ‘Successful’ and ‘Failed’. The other two states, ‘Canceled’ and ‘Suspended’ are due to various other reasons that can also be outside of the data in their website.
So, in order to verify this inference and reject the hypothesis of ‘model isn’t best fit for this dataset’, we applied a binary classification using the same model by omitting the ‘cancelled’ and ‘suspended’ projects. 

### 4.1. Binary Classification using Successful and Failed projects:
In order to verify the above inference, Binomial Logistic Regression is applied again on the dataset by dropping the ‘cancelled’ and ‘suspended’ state projects. The implementation steps remains the same as before. As the data is still imbalanced Oversampling is done to balance the dataset. The data is split in to train and test. The results after testing the model with the test set turned out to be as follows. 

<img width="739" alt="image" src="https://user-images.githubusercontent.com/98278525/193664052-daa36556-8e2b-487b-9fd4-007e2c7aa7ad.png">

The accuracy of the model built is at 97.4% (Fig.16a) with and excellent precision of 0.97 and very good F1 score of 0.97. This specifies that our above inference about the Canceled and Suspended projects, which is, “these states of the projects are there in to that particular state of ‘Canceled’ and ‘Suspended’ for the reasons which are outside the scope of this dataset” is true. 

### 4.2. Conclusion
When the data is tried to model using all the available four classes, the accuracy of the failed class prediction is compromised, whereas the successful class remains with good accuracy. By applying the binary classification on the important classes, we observed the drastic improvement in the models. This leaves us an important learning as follows. When there are multi class classification problems like this and the accuracy of the models cannot be improved, then the problem can be dealt as a binary classification by removing the non-important classes, provided the accuracy of the prediction increases, so that at least two of the important classes can be modeled rather than declaring the dataset as unfit for modeling. 

## 5. References:
Kickstarter (2020), [online] Available at: https://www.kickstarter.com/?ref=nav [Accessed: Feb 21, 2022].

Kickstarter (2015), About — Kickstarter, [online] Kickstarter.com. Available at: https://www.kickstarter.com/about?ref=global-footer [Accessed: 21 Feb, 2022].

www.kaggle.com. (n.d.). Kickstarter Projects, [online] Available at: https://www.kaggle.com/datasets/kemical/kickstarter-projects [Accessed 21 Feb, 2022].

GitHub. (2022). Apache Spark. [online] Available at: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/multiclass_logistic_regression_with_elastic_net.py [Accessed 27 Mar, 2022].

Li, S. (2018). Multi-Class Text Classification with PySpark. [online] Medium. Available at: https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35  [Accessed 21 Feb, 2022].

Brownlee, J. (2020). How to Develop Elastic Net Regression Models in Python. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/elastic-net-regression-in-python/ [Accessed 11 Apr, 2022].

Phatak, M. (2022). Build. [online] GitHub. Available at: https://github.com/phatak-dev/spark-3.0-examples/blob/master/src/main/scala/com/madhukaraphatak/spark/ml/WeightedLogisticRegression.scala [Accessed 11 Apr. 2022].

blog.madhukaraphatak.com. (n.d.). Introduction to Spark 3.0 - Part 4 : Handling Class Imbalance Using Weights. [online] Available at: http://blog.madhukaraphatak.com/spark-3-introduction-part-4/ [Accessed 11 Apr. 2022].

Pulagam, S. (2020). Feature Scaling — Effectively Choose Input Variables Based on Distributions. [online] Medium. Available at: https://towardsdatascience.com/feature-scaling-effectively-choose-input-variables-based-on-distributions-3032207c921f#:~:text=Summary [Accessed 12 Apr. 2022].

Wan, J. (2020). Oversampling and Undersampling with PySpark. [online] Medium. Available at: https://medium.com/@junwan01/oversampling-and-undersampling-with-pyspark-5dbc25cdf253 [Accessed 17 Apr. 2022].


