import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('mint_data.csv')

# Separate customer features
non_actionables = ['Senior Citizen', 'Partner', 'Dependents', 'Tenure Months', 
                   'State Encoded', 'Gender']  

actionables = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
               'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 
               'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']

# Target encode state variable since it has far too many categories
churn_mean_by_state = df.groupby('State')['Churn Value'].mean()
df['State Encoded'] = df['State'].map(churn_mean_by_state)

# One-hot encode remaining non-actionable features
df_encoded = pd.get_dummies(df[non_actionables], drop_first=True)

# Scale features
scaler = StandardScaler()
X_non_actionable_scaled = scaler.fit_transform(df_encoded)

# Use logisitic regression to determine which non-actionable 
# features are most predicive of churn
clf = LogisticRegression(max_iter = 1000)
clf.fit(df_encoded, df['Churn Value'])

coef_df = pd.DataFrame({
    'Feature': df_encoded.columns, 
    'Coefficient': clf.coef_[0]
}).sort_values(by = 'Coefficient', key = abs, ascending = False)

print(coef_df.head())
top_features = coef_df['Feature'].head(3).tolist()
print(top_features)

# Assign groups based on result of logistic regression
df['Group'] = df_encoded[top_features[0]] * 2 + df_encoded[top_features[1]] + 4 * df_encoded[top_features[2]]

# Estimate value at risk per group
groups = []
sizes = []
value_at_risks = []

for i in range(8):
    group = df[df['Group'] == i]
    avg_churn = group["Churn Value"].mean()
    avg_cltv = group["CLTV"].mean()
    
    value_at_risk = avg_churn * avg_cltv
    
    groups.append(i)
    sizes.append(len(group))
    value_at_risks.append(value_at_risk)

# Plot size vs. value at risk
plt.figure(figsize=(8,6))
plt.scatter(sizes, value_at_risks, s=100, c="blue")
plt.xlabel("Size")
plt.ylabel("Value at Risk (per customer)")
plt.title("Size vs Value at Risk by Group")
plt.grid(True)

# Label each point with group number
for i, txt in enumerate(groups):
    plt.annotate(txt, (sizes[i], value_at_risks[i]), textcoords="offset points", xytext=(5,5))

plt.show()

# Use random forest to predict churn value based on service usage
predictor_model = RandomForestClassifier(n_estimators = 100, random_state = 1)

parameter_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [2, 4, 8]
}

grid_search = GridSearchCV(
    estimator = predictor_model,
    param_grid = parameter_grid,
    cv = 4,                
    scoring = 'accuracy',
    verbose = 2
)

selected_groups = [0, 1, 4, 5]

for group in selected_groups:
    # Splitting features/target and test/train
    X = df[df['Group'] == group][actionables]
    y = df[df['Group'] == group]['Churn Value']

    X_encoded = pd.get_dummies(X, drop_first = True).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2, random_state = 1)
    
    # Fitting random forest
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(test_score)

    # Interpret results using Shapley Values
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[:, :, 1], X_test)

    mean_shap_by_group = pd.Series(abs(shap_values[:, :, 1]).mean(axis = 0), index = X_train.columns)
    mean_shap_by_group = mean_shap_by_group.sort_values(ascending=False).reset_index()
    mean_shap_by_group.columns = ["Feature", "Mean_SHAP"]
    mean_shap_by_group["Group"] = group
    print(mean_shap_by_group.head())
