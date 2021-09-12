# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 22:49:01 2021

@author: Nicolas Ponte
@updated by: Group D:
• 	Ahmed Aljeshi
•	Juan Francisco Balbi 
•	Paula Escusol Entío
•	Isobel Rae Impas 
•	Paliz Mungkaladung
    
"""
"""
list of vectorized functions used:
    - pd.isnull()
    - pd.where()
    - pd.sum()
    - all sklearn functions


"""

import pandas as pd
import numpy as np



class RiskDataframe(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes to a prticular
    type of Dataframes in the Risk Indistry. 
    It provides the end user with both general and specific cleaning functions, 
    though they never reference a specific VARIABLE NAME.
    
    It facilitates the End User to perform some Date Feature Engineering,
    Scaling, Encoding, etc. to avoid code repetition.
    """

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = RiskDataframe(*args,**kwargs)
            return df
        return func_
    

#-----------------------------------------------------------------------------
                        # RISK BASED APPROACH
#-----------------------------------------------------------------------------    
    def missing_not_at_random(self, *args):
       
        """
             
        Parameters
        ----------
        self   : TYPE: Dataframe
            DESCRIPTION: The dataframe with the data to be analyzed
        Returns
        -------
        A print with the analysis of the missing not at random status of the Dataframe provided.
        """
        
        # Checking for user erros
        
        if len(args) > 0:
            print ('More than one variable was passed.  missing_not_at_random accepts only one variable')
            return()
        
        if not isinstance(self, pd.DataFrame):
            print ('Wrong Arguments were passed')
            print ('the correct format is: RiskDataframe.missing_not_at_random(DataFrame) where DataFrame is your DataFrame Name')
            return()
       
        mnar_columns = []
        test_columns =[]
        related_columns = []
        related_categories = []
        
        null_columns=self.columns[self.isnull().any()]
        
        # Initializing test columns to include 0's where the corresponding category is null and 1 otherwise
        for i in null_columns:
          self[i + '_test'] = np.where(self[i].isnull(),0, 1)
          test_columns.append(i + '_test')
         
        categorical_variables =  self.select_dtypes(exclude=np.number).columns
     
        
         # A nested for loop to pupulate:
         #    1. The mnar_columns: includes column names with values that mnar_columns
         #    2. related_columns: includes the column names related to the mnar_columns
         #    3. related_categories: related categories to mnar_columns
             
        for t in test_columns:
          
          for col in categorical_variables:
              
            
            for cat in self[col].unique():
          
              not_missing = self[self[col] == cat][t].sum()
           
              all_cat_rows = self[self[col] == cat].shape[0]
              
              if all_cat_rows > 0:
                  condition = (all_cat_rows - not_missing)/all_cat_rows
              else:
                  condition = 0
                 
      
              if condition > 0.9:
                  mnar_columns.append(t[:len(t) - 5])
                  related_columns.append(col)
                  related_categories.append(cat)
        
            
        self.drop(test_columns, axis=1, inplace = True)
        
        # Printing the MNAR Report
        
        thin_columns = [x for x in self.columns if x not in mnar_columns]
               
        print ('\nMissing Not at Random Report - ', mnar_columns, 'variables seem Missing Not at Random, there for we recommend:')
        print ('\n\n   Thin File Segment Variables: ', thin_columns)
        print ('\n\n   Full File Segment Variables: ', self.columns)
        print ('\n\nPlease note that they are related to the Variable(s):', related_columns, ' and category(s):', related_categories, ', respectively')
          
        
        
         
    
 
    def find_segment_split(self, target, all_variables, observation_rate, *args):
        """
       
        Returns
        -------
        Example 1: ACCOUNT_NUMBER Not good for segmentation. Afer analysis, we did not find a good split using this variable.
        Example 2: SEX Good for segmentation.  +
                Segment1: SEX in ('F') [Accuracy Full Model: 32% / Accuracy Segmented Model: 33%]
                Segment2: SEX in ('M') [Accuracy Full Model: 63% / Accuracy Segmented Model: 68%]
        -------
        Parameters
        ----------
        self   : TYPE: Dataframe
            DESCRIPTION: The dataframe with the data to be analyzed
        target : TYPE: string
            DESCRIPTION: contains the target variable
        all_variables : TYPE: list
            DESCRIPTION: list of variables to be considered for analysis
        observation_rate : TYPE: dictionary
            DESCRIPTION:dictionary of the categorical variables observation rate 
   
        """
        
        # Checking for user errors
        
        if len(args) > 0:
          print ('More than four variable were passed.  find_segment_split accepts only four variables')
          print ('Where:\n   - DataFrame is the name of your dataframe')
          print ('   - target is the name of your target variable')
          print ('   - all_variables is the list of the variables to be included in the analysis')
          print ('   - observation_rate is a dictionary that includes the observation rate of each category in categorical variables')
          return()
      
        if not isinstance(self, pd.DataFrame):
            print ('Wrong Dataframe name was passed')
            print ('the correct format is: RiskDataframe.find_segment_split(DataFrame, target, all_variables[], observation_rate[])')
            print ('Where:\n   - DataFrame is the name of your dataframe')
            print ('   - target is the name of your target variable')
            print ('   - all_variables is the list of the variables to be included in the analysis')
            print ('   - observation_rate is a dictionary that includes the observation rate of each category in categorical variables')
            return()
        try:
            if not all(isinstance(self[element].iat[0], (int, float)) for element in all_variables):
                print ('The list of variables provided contains non-numeric variables')
                return()
        except: 
            print ('Wrong list was passed, please make sure that all the variables in the list are included in the Dataframe')
            return()
                
        # To ensure robustness, we scaled all the variables using MinMax Scaler

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        for v in all_variables:
            self[v+'_original'] = self[v]
            self[v] = scaler.fit_transform(self[[v]])
    
        # Splitting the Dataset
        
        from sklearn.model_selection import train_test_split
        splitter = train_test_split
        "-----------------------"
        
        df_train, df_test = splitter(self, test_size = 0.2, random_state = 42)
        
        try:
            X_train = df_train[all_variables]
        except:
            print ('Wrong variable list was passed, please make sure that you pass the right variables and that all of them are included in your dataframe')
            return()
        
        try:
            y_train = df_train[target]
        except:
            print ('Wrong target variable was passed, please make sure that you pass the right target and that it is included in your dataframe')
            return()
      
            
        
        # Running the full model on all variables
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedKFold
        
        method = LogisticRegression(random_state=0)
        fitted_full_model = method.fit(X_train, y_train)
        

        
      
        #running decision trees to decide on which variables to use for segmentation and their threshold
        
        from sklearn import tree
        
        X = df_train[all_variables]
        Y = df_train[target]
        
        #build decision tree
        clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=4,min_samples_leaf=4)
        #max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node
        
        #fit the tree to iris dataset
        clf.fit(X,Y)
        
        # Function to get the Threshold
        # Reference: https://mljar.com/blog/extract-rules-decision-tree/ (with modification)
        
        from sklearn.tree import export_text
        from sklearn.tree import _tree
       
        tree_rules = export_text(clf, feature_names=list(X.columns))
        
        def get_thresholds(tree, feature_names, class_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
        
            paths = []
            path = []
            
            final_thresholds = {}
        
            def recurse(node, path, paths):
                
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    p1, p2 = list(path), list(path)
                    p1 += [f"({name} <= {np.round(threshold, 3)})"]
                    recurse(tree_.children_left[node], p1, paths)
                    p2 += [f"({name} > {np.round(threshold, 3)})"]
                    recurse(tree_.children_right[node], p2, paths)
                    final_thresholds[name] = threshold
                else:
                    path += [(tree_.value[node], tree_.n_node_samples[node])]
                    paths += [path]
                return final_thresholds
        
            final_thresholds = recurse(0, path, paths)
            
            """
            # Was included in the original function
            # sort by samples count
            samples_count = [p[-1][1] for p in paths]
            ii = list(np.argsort(samples_count))
            paths = [paths[i] for i in reversed(ii)]
            
            rules = []
            for path in paths:
                rule = "if "
                
                for p in path[:-1]:
                    if rule != "if ":
                        rule += " and "
                    rule += str(p)
                rule += " then "
                if class_names is None:
                    rule += "response: "+str(np.round(path[-1][0][0][0],3))
                else:
                    classes = path[-1][0][0]
                    l = np.argmax(classes)
                    rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                rule += f" | based on {path[-1][1]:,} samples"
                rules += [rule]
            """   
            
            return final_thresholds
            
            
        final_thresholds = get_thresholds(clf, all_variables, all_variables)
        
        relevant_columns = []
        relevant_categories = []
        
        # Function to get the releveant columns and relevant categorical columns
        
        def column_types(variable, relevant_columns, relevant_categories):
            
          
            if variable in tree_rules:
                relevant_columns.append(variable)
                if variable[:len(variable) - 5] in self.select_dtypes(exclude=np.number).columns:
                    relevant_categories.append(variable[:len(variable) - 5])
            
        
        for i in X.columns:
            column_types(i, relevant_columns, relevant_categories)
        
        
        
        category_segments = {}
        
        # A for loop to get the categorical segments according their repective observation ratio threshold
        
        for i in relevant_categories:
          column_header = i+'_rate'
          try:
              for cat in observation_rate[i]:   
                if observation_rate[i][cat] >= final_thresholds[column_header]: 
                  first_segment = i+'_segment1'
                  category_segments.setdefault(first_segment, []).append(cat)
                else:  
                  second_segment = i+'_segment2'
                  category_segments.setdefault(second_segment, []).append(cat)
          except:
              print ('Wrong observation_rate dictionary was passed.')
              print ('Please make sure that you pass the dictionary and that it includes the observation rate for all categorical variables')
              return()
        

             
        print ('\nVariable by Variable Risk Based Segmentation Analysis:\n')
        for variable in all_variables:
            
            
            
            if variable in relevant_columns:
                # Running the model on each segment and comparing the output using the GINI metric
                
                df_train_seg1 = df_train[self[variable] <final_thresholds[variable]]
                df_train_seg2 = df_train[self[variable] >final_thresholds[variable]]
                df_test_seg1 = df_test[self[variable] <final_thresholds[variable]]
                df_test_seg2 = df_test[self[variable] >final_thresholds[variable]]
        
                X_train_seg1 = df_train_seg1[all_variables]
                y_train_seg1 = df_train_seg1[target]
                X_test_seg1 = df_test_seg1[all_variables]
                y_test_seg1 = df_test_seg1[target]
                fitted_model_seg1 = method.fit(X_train_seg1, y_train_seg1)
                
                X_train_seg2 = df_train_seg2[all_variables]
                y_train_seg2 = df_train_seg2[target]
                X_test_seg2 = df_test_seg2[all_variables]
                y_test_seg2 = df_test_seg2[target]
                fitted_model_seg2 = method.fit(X_train_seg2, y_train_seg2)        
                
                            
                 
                # A function to get the GINI score
                
                def GINI(y_test, y_pred_probadbility):
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probadbility)
                    roc_auc = auc(fpr, tpr)
                    GINI = (2 * roc_auc) - 1
                    return(GINI)
 
                y_pred_seg1_proba = fitted_model_seg1.predict_proba(X_test_seg1)[:,1]
                y_pred_seg1_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg1)[:,1]
                y_pred_seg2_proba = fitted_model_seg2.predict_proba(X_test_seg2)[:,1]
                y_pred_seg2_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg2)[:,1]
                
                           
                
                if variable[:len(variable) - 5] in self.select_dtypes(exclude=np.number).columns:
                    original_variable = variable[:len(variable) - 5]
                    print ("\n  ", original_variable, "- Good for segmentation:")
            
                    print("\n     Segment1:", original_variable, "in", category_segments[original_variable+'_segment1'], "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg1, y_pred_seg1_proba)*100,
                        GINI(y_test_seg1, y_pred_seg1_fullmodel_proba)*100
                    )) 
            
                    print("\n     Segment2:", original_variable, "in", category_segments[original_variable+'_segment2'], "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg2, y_pred_seg2_proba)*100,
                        GINI(y_test_seg2, y_pred_seg2_fullmodel_proba)*100
                    )) 
                else:
                    
                    
                    print ("\n  ", variable, "- Good for segmentation:")
                    
                    print("\n     Segment1:", variable, "<", self[self[variable] > final_thresholds[variable]][variable+'_original'].min(), "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg1, y_pred_seg1_proba)*100,
                        GINI(y_test_seg1, y_pred_seg1_fullmodel_proba)*100
                    )) 
            
                    print("\n     Segment2:", variable, ">", self[self[variable] > final_thresholds[variable]][variable+'_original'].min(), "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg2, y_pred_seg2_proba)*100,
                        GINI(y_test_seg2, y_pred_seg2_fullmodel_proba)*100
                    )) 
            
                    
            else: 
              print ("\n", variable, "is not good for segmentation. After analysis we did not find a good split using this variable") 
                    
            if (variable+'_original') in self.select_dtypes(include=np.number).columns:
                  self[variable] = self[variable+'_original']
                  self.drop([variable+'_original'], axis = 1, inplace=True)  
                          
    def final_report(target, mnar_report, split_report, *args):
        """
        This method combines the mnar_report and the splite report
        

        Parameters
        ----------
        target : TYPE: string
            DESCRIPTION: contains the target variable
        mnar_report : TYPE: string
            DESCRIPTION: contains the output of the missing_not_at_random method
        split_report : TYPE: string
            DESCRIPTION:contains the output of the find_segment_split method
   

        Returns
        -------
        A print of the combined report

        """
        if len(args) > 0:
          print ('More than three variable were passed.') 
          print ('final_report accepts only three string variables (target, mnar_report, split_report)')
        
              
        try:   
            final_report = 'Execution Summary Report' + ':\n   ' + target + 'is the target variable and was not analized separetly.\n' + mnar_report + '\n' + split_report
            print (final_report)
        except:
            print ('Wrong Arguments were passed, please make sure you pass thee string arguments')
            