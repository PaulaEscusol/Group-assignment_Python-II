# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 22:49:01 2021

@author: Nicolas Ponte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

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
                        # DATA HANDLING
#-----------------------------------------------------------------------------

    def SetAttributes(self, kwargs):
        """
        The function will update the type of the variable submitted for change.
        It will veify first that the key is present in the desired dataframe.
        If present, it will try to change the type to the desired format.
        If not possible, it will continue to the next element.         
        Parameters
        ----------
        **kwargs : The key-argument pair of field-type relationship that
        wants to be updated.
        Returns
        -------
        None.
        """
        if self.shape[0] > 0:
            for key,vartype in kwargs.items():
                if key in self.columns:
                    try:
                        self[key] = self[key].astype(vartype)
                    except:
                        print("Undefined type {}".format(str(vartype)))
                else:
                    print("The dataframe does not contain variable {}.".format(str(key)))
        else:
            print("The dataframe has not yet been initialized")

#-----------------------------------------------------------------------------
                        # RISK BASED APPROACH
#-----------------------------------------------------------------------------    
    def missing_not_at_random(self, input_vars=[] ):
        """

        Returns
        -------
        A print with the analysis.

        """
        
       
        mnar_columns = []
        test_columns =[]
        null_columns=self.columns[self.isnull().any()]
        for i in null_columns:
          self[i + '_test'] = np.where(self[i].isnull(),0, 1)
          test_columns.append(i + '_test')
         
        categorical_variables =  self.select_dtypes(exclude=np.number).columns
        
        for t in test_columns:
          
          for col in categorical_variables:
            for i in self[col].unique():
          
              not_missing = self[self[col] == i][t].sum()
           
              test_rows = self[self[col] == i].shape[0]
           
              condition = (test_rows - not_missing)/test_rows
      
              if condition > 0.9:
                  mnar_columns.append(t)
        
        
        
        full = []

        for i in mnar_columns:
            full.append(i[:len(i) - 5])
        
        self.drop(test_columns, axis=1, inplace = True)
        full_mnar_columns = full + mnar_columns
        thin_columns = [x for x in self.columns if x not in full_mnar_columns]
        
       
        
        
        print ('Missing Not at Random Report - ', full, 'variables seem Missing Not at Random, there for we recommend:')
        print ('\n\nThin File Segment Variables: ', thin_columns)
        print ('\n\nFull File Segment Variables: ', self.columns)
          
        
        
         
    
 
    def find_segment_split(self, observation_rate):
        """
        find_segment_split(self, canditate='', input_vars=[], target='' )
        Returns
        -------
        Example 1: ACCOUNT_NUMBER Not good for segmentation. Afer analysis, we did not find a good split using this variable.
        Example 2: SEX Good for segmentation.  +
                Segment1: SEX in ('F') [Accuracy Full Model: 32% / Accuracy Segmented Model: 33%]
                Segment2: SEX in ('M') [Accuracy Full Model: 63% / Accuracy Segmented Model: 68%]
                
        """
        from sklearn import tree
        from matplotlib import pyplot as plt
        
        target = 'bucket'
        all_variables = ['original_booked_amount', 'outstanding', 'car_type_rate', 'age', 'loan_expected_duration', 'program_name_rate', 'profession_rate', 'sex']
        categorical_variables = self.select_dtypes(exclude=np.number).columns
        
#running full model
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        splitter = train_test_split
        "-----------------------"
        
        df_train, df_test = splitter(self, test_size = 0.2, random_state = 42)
        
        
        X_train = df_train[all_variables]
        y_train = df_train[target]
        
        X_test = df_test[all_variables]
        y_test = df_test[target]
        
        from sklearn.linear_model import LogisticRegression
        method = LogisticRegression(random_state=0)
        fitted_full_model = method.fit(X_train, y_train)
        y_pred = fitted_full_model.predict(X_test)
        
        y_pred = fitted_full_model.predict_proba(X_test)[:,0]
        
        #GINI Coefficient
        from sklearn.metrics import roc_curve, auc
        fpr,tpr,thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr,tpr)
        GINI = (2*roc_auc) -1
        
      
#running decision trees
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        splitter = train_test_split
        "-----------------------"
        
        df_train, df_test = splitter(self, test_size = 0.2, random_state = 42)
        
        X = df_train[all_variables]
        Y = df_train[target]
        
        #build decision tree
        clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=4,min_samples_leaf=4)
        #max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node
        
        #fit the tree to iris dataset
        clf.fit(X,Y)
        
        # Reference: https://mljar.com/blog/extract-rules-decision-tree/ (with modification)
        from sklearn.tree import export_text
        from sklearn.tree import _tree
        #tree_rules = export_text(clf, feature_names=list(X.columns))
        tree_rules = export_text(clf, feature_names=list(X.columns))
        
        def get_rules(tree, feature_names, class_names):
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
                
            return rules, final_thresholds

        rules, final_thresholds = get_rules(clf, all_variables, all_variables)
        
        
        
          
        relevant_columns = []
        relevant_categories = []
       
        
        for variable in X.columns:
            if variable in tree_rules:
                relevant_columns.append(variable)
                if variable[:len(variable) - 5] in self.select_dtypes(exclude=np.number).columns:
                    relevant_categories.append(variable[:len(variable) - 5])
       #     else:
       #        print (variable, ": Not good for segmentation. After analysis we did not find a good split using this variable")
        
        category_segments = {}

        for i in relevant_categories:
          column_header = i+'_rate'
          for cat in observation_rate[i]:   
            if observation_rate[i][cat] >= final_thresholds[column_header]: 
              first_segment = i+'_segment1'
              category_segments.setdefault(first_segment, []).append(cat)
            else:  
              second_segment = i+'_segment2'
              category_segments.setdefault(second_segment, []).append(cat)
        
        for variable in X.columns:
            if variable in relevant_columns:
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
                    print ("\n", original_variable, "- Good for segmentation:")
            
                    print("\n     Segment1:", original_variable, "in", category_segments[original_variable+'_segment1'], "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg1, y_pred_seg1_proba)*100,
                        GINI(y_test_seg1, y_pred_seg1_fullmodel_proba)*100
                    )) 
            
                    print("\n     Segment2:", original_variable, "in", category_segments[original_variable+'_segment2'], "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg2, y_pred_seg2_proba)*100,
                        GINI(y_test_seg2, y_pred_seg2_fullmodel_proba)*100
                    )) 
                else:
                    
                    print ("\n", variable, "- Good for segmentation:")

                    print("\n     Segment1:", variable, "<", final_thresholds[variable], "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg1, y_pred_seg1_proba)*100,
                        GINI(y_test_seg1, y_pred_seg1_fullmodel_proba)*100
                    )) 
            
                    print("\n     Segment2:", variable, ">", final_thresholds[variable], "[GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(
                        GINI(y_test_seg2, y_pred_seg2_proba)*100,
                        GINI(y_test_seg2, y_pred_seg2_fullmodel_proba)*100
                    )) 
            
                    
            else: 
              print ("\n", variable, "Not good for segmentation. After analysis we did not find a good split using this variable") 
                    
            
                            
                    