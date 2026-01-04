import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(
    page_title="📊 Train Your Model App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"  
)

st.markdown("""
<div style="background-color:#1f1f1f; padding: 15px; border-radius: 12px; color:white;">
<h1 style="color:#00bfff; text-align:center;">🤖 Train Your Model By Your Way</h1>
<p style="text-align:center; font-size:16px; color:white;">
Upload your dataset and experiment with different machine learning models for <b>Regression</b> and <b>Classification</b>. 
Visualize predictions, measure accuracy, and explore your data easily! 🌟
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style="background-color:#2e2e2e; padding: 10px; border-radius: 8px; color:white;">
<h3>📝 Instructions:</h3>
<ul>
<li>Upload your CSV file. 📁</li>
<li>Choose the target column for either Regression or Classification. 🎯</li>
<li>Select features to include or drop unnecessary columns. ❌</li>
<li>Pick a model and adjust its hyperparameters if needed. ⚙️</li>
<li>View metrics and interactive charts to analyze results. 📈</li>
</ul>
</div>
""", unsafe_allow_html=True)

data = st.file_uploader("Upload Your File Here 📁", type="csv", key="file_uploader")
st.caption("Choose a CSV file from your computer to start analyzing your data.")

if data is not None:
    data = pd.read_csv(data)
    data = data.reset_index(drop=True)

    n_row = st.slider("Enter the number of rows to present 📄",
                      min_value=5, max_value=data.shape[0], key="row_slider")
    st.caption("Select how many rows you want to preview in the table.")
    
    choosed_column = st.multiselect("Choose the columns to display 📊",
                                    options=data.columns, default=data.columns, key="multi_1")
    st.caption("Select which columns to display in the preview table.")
    st.write(data.loc[0:n_row, choosed_column])

    hated_columns = st.multiselect("Choose the columns you want to drop ❌",
                                   options=data.columns, key="multi_2")
    st.caption("Remove unnecessary columns before training the model.")
    data.drop(hated_columns, axis="columns", inplace=True)

    st.subheader("The data after deleting")
    st.write(data.head(10))

    st.subheader("Target 🎯")
    tab1, tab2 = st.tabs(["Regression 📈", "Classification 🤖"])

    with tab1:
        numerical_cols = data.select_dtypes(include="number").columns
        if len(numerical_cols) == 0:
            st.warning("⚠️ No Numerical columns found. Regression cannot be performed.")
        else:
            target = st.selectbox("Choose the target column (numeric) 🎯",
                                  options=data.select_dtypes(include="number").columns, key="target_reg")
            st.caption("Pick a numeric column as the target variable for regression.")
            sim_target = SimpleImputer(strategy="mean")
            data[target] = sim_target.fit_transform(data[[target]])[:, 0]
            y = data[target]
            X = data.drop(target, axis="columns")
    
            st.subheader("Final train data")
            st.write(X.head())
    
            numerical_columns = X.select_dtypes(include="number").columns
            categorical_columns = X.select_dtypes(include="object").columns

            numerical_pipe = Pipeline(steps=[
                ("number_prerocessing_mean", SimpleImputer(strategy="mean")),
                ("number_prerocessing_standard", StandardScaler())
            ])
            categorical_pipe = Pipeline(steps=[
                ("categrical_frequant", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder())  
            ])

            st.subheader("The final shape of the data")
            preprocissing_data = None
            if len(numerical_columns) > 0 and len(categorical_columns) > 0:
                full_transform = ColumnTransformer(transformers=[
                    ("numerical", numerical_pipe, numerical_columns),
                    ("categorical", categorical_pipe, categorical_columns)
                ])
                preprocissing_data = full_transform.fit_transform(X)
                if hasattr(preprocissing_data, "toarray"):
                    preprocissing_data = preprocissing_data.toarray()
                preprocissing_data = pd.DataFrame(preprocissing_data, columns=full_transform.get_feature_names_out())
                st.write(preprocissing_data.head(10))
            elif len(numerical_columns) > 0:
                partial_transform_1 = ColumnTransformer(transformers=[
                    ("numerical", numerical_pipe, numerical_columns),
                ])    
                preprocissing_data = partial_transform_1.fit_transform(X)
                if hasattr(preprocissing_data, "toarray"):
                    preprocissing_data = preprocissing_data.toarray()
                preprocissing_data = pd.DataFrame(preprocissing_data, columns=partial_transform_1.get_feature_names_out())
                st.write(preprocissing_data.head(10))
            elif len(categorical_columns) > 0:
                partial_transform_2 = ColumnTransformer(transformers=[
                    ("categorical", categorical_pipe, categorical_columns) 
                ])        
                preprocissing_data = partial_transform_2.fit_transform(X)
                if hasattr(preprocissing_data, "toarray"):
                    preprocissing_data = preprocissing_data.toarray()
                preprocissing_data = pd.DataFrame(preprocissing_data, columns=partial_transform_2.get_feature_names_out())
                st.write(preprocissing_data.head(10))
            else:
                st.warning("The columns are not enough to train the data")
    
            model_choosed = st.selectbox("Choose regression model 🤖",
                                         options=["Linear_Regression", "Random_Forest", "XGBOOST_reg", "Gradient Boost reg"],
                                         key="model_reg")
            st.caption("Select which regression model you want to train on your data.")
    
            ratio = st.slider("Choose test data percentage 📊", min_value=5, max_value=95, key="split_ratio_reg")
            st.caption("Select the percentage of data to be used as test set.")
            train_data, test_data, train_target, test_target = train_test_split(preprocissing_data, y, test_size=ratio/100, shuffle=True)
    
            if model_choosed == "Linear_Regression":
                model = LinearRegression()
                model.fit(train_data, train_target)
                predict_values = model.predict(test_data)

            elif model_choosed == "Gradient Boost reg":
                n_estimators = st.number_input("n_estimators 🔢", min_value=50, max_value=1000, value=100, step=50, key="gb_n")
                learning_rate = st.slider("learning_rate ⚡", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="gb_lr")
                max_depth = st.selectbox("max_depth 📏", options=[2, 3, 5, 7, 10], key="gb_depth")
                subsample = st.slider("subsample 🌐", min_value=0.5, max_value=1.0, value=1.0, step=0.1, key="gb_sub")
                criterion = st.selectbox("criterion 📝", options=["squared_error", "friedman_mse", "absolute_error", "poisson"], key="gb_cri")
                loss = st.selectbox("loss 🔧", options=["squared_error", "absolute_error", "huber", "quantile"], key="gb_loss")
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    criterion=criterion,
                    loss=loss,
                    random_state=42
                )
                model.fit(train_data, train_target)
                predict_values = model.predict(test_data)

            elif model_choosed == "XGBOOST_reg":
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                model.fit(train_data, train_target)
                predict_values = model.predict(test_data)
    
            elif model_choosed == "Random_Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(train_data, train_target)
                predict_values = model.predict(test_data)
                      
            choosed_accuracy = st.selectbox("Choose regression metric 📈",
                                            options=["mean_absolute_error", "mean_squared_error", "r2_score"],
                                            key="acc_reg")
            st.caption("Pick the metric to evaluate your regression model.")

            if choosed_accuracy == "mean_absolute_error":
                st.subheader("MAE:")
                st.write(mean_absolute_error(test_target, predict_values))
            elif choosed_accuracy == "mean_squared_error":
                st.subheader("MSE:")
                st.write(mean_squared_error(test_target, predict_values))
            else:
                st.subheader("R2 Score:")
                st.write(r2_score(test_target, predict_values))
    
            # visualize
            x_axis = np.arange(0, len(test_target))
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=predict_values, mode="lines+markers", name="Predictions"))
            fig.add_trace(go.Scatter(x=x_axis, y=test_target, mode="lines+markers", name="Actual"))
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------------



    with tab2:

        categorical_cols = data.select_dtypes(include="object").columns
        if len(categorical_cols) == 0:
            st.warning("⚠️ No categorical columns found. Classification cannot be performed.")
        else:
            target = st.selectbox("Choose the target column (categorical) 🎯",
                                  options=data.select_dtypes(include="object").columns, key="target_clf")
            st.caption("Pick a categorical column as the target variable for classification.")
    
            sim_target = SimpleImputer(strategy="most_frequent")
            data[target] = sim_target.fit_transform(data[[target]])[:, 0]
            y = data[target]
    
            if y.nunique() > 20:
                st.error("❌ The column you selected as target has too many unique values. Please choose another column suitable for classification.")
            else:
                X = data.drop(target, axis="columns")
                st.subheader("Final train data")
                st.write(X.head())
    
                numerical_columns = X.select_dtypes(include="number").columns
                categorical_columns = X.select_dtypes(include="object").columns
    
                numerical_pipe = Pipeline(steps=[
                    ("number_prerocessing_mean", SimpleImputer(strategy="mean")),
                    ("number_prerocessing_standard", StandardScaler())
                ])
                categorical_pipe = Pipeline(steps=[
                    ("categrical_frequant", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore"))
                ])
    
                st.subheader("The final shape of the data")
                preprocissing_data = None
                if len(numerical_columns) > 0 and len(categorical_columns) > 0:
                    full_transform = ColumnTransformer(transformers=[
                        ("numerical", numerical_pipe, numerical_columns),
                        ("categorical", categorical_pipe, categorical_columns)
                    ])
                    preprocissing_data = full_transform.fit_transform(X)
                    if hasattr(preprocissing_data, "toarray"):
                        preprocissing_data = preprocissing_data.toarray()
                    preprocissing_data = pd.DataFrame(preprocissing_data, columns=full_transform.get_feature_names_out())
                    st.write(preprocissing_data.head(10))
                elif len(numerical_columns) > 0:
                    partial_transform_1 = ColumnTransformer(transformers=[
                        ("numerical", numerical_pipe, numerical_columns),
                    ])
                    preprocissing_data = partial_transform_1.fit_transform(X)
                    if hasattr(preprocissing_data, "toarray"):
                        preprocissing_data = preprocissing_data.toarray()
                    preprocissing_data = pd.DataFrame(preprocissing_data, columns=partial_transform_1.get_feature_names_out())
                    st.write(preprocissing_data.head(10))
                elif len(categorical_columns) > 0:
                    partial_transform_2 = ColumnTransformer(transformers=[
                        ("categorical", categorical_pipe, categorical_columns)
                    ])
                    preprocissing_data = partial_transform_2.fit_transform(X)
                    if hasattr(preprocissing_data, "toarray"):
                        preprocissing_data = preprocissing_data.toarray()
                    preprocissing_data = pd.DataFrame(preprocissing_data, columns=partial_transform_2.get_feature_names_out())
                    st.write(preprocissing_data.head(10))
                else:
                    st.warning("The columns are not enough to train the data")
    
                model_choosed = st.selectbox("Choose classification model 🤖",
                                             options=["Logistic_Regression", "Random_Forest_Classifier", "XGBOOST_Classifier"],
                                             key="model_clf")
                st.caption("Pick the classification model to train on your dataset.")
    
                ratio = st.slider("Choose test data percentage 📊", min_value=5, max_value=95, key="split_ratio_clf")
                st.caption("Select the percentage of data to be used as test set for classification.")
                train_data, test_data, train_target, test_target = train_test_split(preprocissing_data, y, test_size=ratio/100, shuffle=True)
    
                if model_choosed == "Logistic_Regression":
                    from sklearn.linear_model import LogisticRegression
                    C = st.slider("Inverse of regularization (C)", 0.01, 10.0, 1.0, 0.01, key="log_c")
                    penalty = st.selectbox("Penalty", ["l2", "none"], key="log_pen")
                    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key="log_solver")
                    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)
    
                elif model_choosed == "Random_Forest_Classifier":
                    from sklearn.ensemble import RandomForestClassifier
                    n_estimators = st.number_input("n_estimators", 50, 1000, 100, step=50, key="rf_n")
                    max_depth = st.selectbox("max_depth", [None, 5, 10, 20], key="rf_depth")
                    criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"], key="rf_cri")
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        criterion=criterion,
                        random_state=42
                    )
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)
    
                elif model_choosed == "XGBOOST_Classifier":
                    from xgboost import XGBClassifier
                    n_estimators = st.number_input("n_estimators", 50, 1000, 100, step=50, key="xgb_n")
                    learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01, key="xgb_lr")
                    max_depth = st.selectbox("max_depth", [2, 3, 5, 7, 10], key="xgb_depth")
                    subsample = st.slider("subsample", 0.5, 1.0, 1.0, 0.1, key="xgb_sub")
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        use_label_encoder=False,
                        eval_metric="mlogloss",
                        random_state=42
                    )
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)
    
                from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
    
                choosed_accuracy = st.multiselect("Choose classification metrics 📊",
                                                  options=["accuracy_score", "f1_score", "precision_score", "confusion_matrix"],
                                                  key="acc_clf")
                st.caption("Pick metrics to evaluate your classification model.")
    
                if choosed_accuracy:
                    for metric in choosed_accuracy:
                        if metric == "accuracy_score":
                            st.subheader("Accuracy:")
                            st.write(accuracy_score(test_target, predict_values))
                        elif metric == "f1_score":
                            st.subheader("F1 Score:")
                            st.write(f1_score(test_target, predict_values, average="weighted"))
                        elif metric == "precision_score":
                            st.subheader("Precision:")
                            st.write(precision_score(test_target, predict_values, average="weighted"))
                        elif metric == "confusion_matrix":
                            st.subheader("Confusion Matrix:")
                            cm = confusion_matrix(test_target, predict_values)
                            fig = px.imshow(
                                cm,
                                x=[f"Pred {cls}" for cls in np.unique(test_target)],
                                y=[f"True {cls}" for cls in np.unique(test_target)],
                                text_auto=True,
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig)
       
