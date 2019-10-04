# Predicting Absenteeism In WorkPlace with LogisticRegression

in this project i Analyze data from the file "Absenteeism-data.csv" 
and predict the probailty that an employee will be absent from work for more than the median of absent hours


the data contains the columns: ID	Reason for Absence	Date	Transportation Expense	Distance to Work	Age	Daily Work Load Average	Body Mass Index	Education	Children	Pets	Absenteeism Time in Hours

# Analyze and Edit the Data

im handling the Reasons for Absence and grouping them to 4 groups and creating Dummies,

changing the Date column and extracting day of the week and month values,

handling the Education by mapping to High school Dipolma or Higher(0 for high school),

Standazring the Data by using a Custom Scaler to scale only the columns that i want(and not the dummies)-CustomScaler.py

Dropping some not that usefull data to simplify the model.



# Machine Learning by LogisticRegression 

Splitting the data to Train and test.
Traning the Model and printing a Summary Table for the intercept and Coefficient.
Printing out the accuracy of the model.


# Saving the Model to Module

im saving the model to 'Model', Scaler to 'Scaler'.
Saving the Model that i created to the file 'absenteeism_module.py' and making a file 'MainUsingTheModule.py' to use the module with new data and that it will be ready for deployment






Built with SKlearn,pandas,numpy
