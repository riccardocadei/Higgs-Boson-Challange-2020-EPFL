# higgs_boson_classification
CS-433

PROJECT TO DO:

1. Check **twice** the file `run.py` that we will submit. We need to chceck that no external libraries is used, the names are correct, the output are correct (correct output: **DONE**).
2. Remove the normalization term in the `compute_loss` for logistic. **DONE**
3. Print graphs comparing all the methods for the final report.
4. Correct file name! Its not clear if the file `implementations.py` should contains only the 6 methods, or the 6 methods + all their necessary methods (I would think the second one).


TO DO:

1. Preprocessing:

    1.1 Feature Selections: 

        STATISTICAL FEATURES ANALYSIS
        New features: Apply a polynomial basis to all the X features
        PCA, correlation analysis (scatterplot, VIF, ...), manage the 0s in the last feature

    1.2. Outlyer: 

        Outlayer analysis, leverages, cook's metric ...

    1.3. Missing Values:
    
        Define a new Metric and sobstitute the missing values with the rispective value of the most similar training exampple
      
2. Cross Validation for each model like cross_validation_least_squares_GD and cross_validation_ridge_regression (4 remaining)

3. Cross Validation to find Optimal Lambda in model Ridge and Reg_Logistic_Reg


PLUS:

4. Prepare and share the template for the report (in overleaf we already have a template that they give us)

5. Load the test.csv zipped on gitHub and modify the function load_csv in order to unzip directly this file




status: Accuracy = 0.782

GOAL1: Accuracy >0.80
GOAL2: Accuracy >0.83



