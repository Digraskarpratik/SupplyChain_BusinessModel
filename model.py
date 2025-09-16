# Import Data Manipulation Library
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Multicolinearity test and treatment libraries
from sklearn.decomposition import PCA

url = "https://raw.githubusercontent.com/Digraskarpratik/SupplyChain_BusinessModel/refs/heads/main/SCM.csv"

df = pd.read_csv(url)

df.drop(["wh_est_year", "workers_num", "approved_wh_govt_certificate", "Ware_house_ID", "WH_Manager_ID", "WH_regional_zone"], axis=1, inplace=True)


from collections import OrderedDict

stats = []

for col in df.columns:
    if df[col].dtype != "object":
        numerical_stats = OrderedDict({
           'Feature': col,
            'Minimum': df[col].min(),
            'Maximum': df[col].max(),
            'Mean': df[col].mean(),
            "median" : df[col].median(),
            'Mode': df[col].mode()[0] if not df[col].mode().empty else None,
            '25%': df[col].quantile(0.25),
            '75%': df[col].quantile(0.75),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'Standard Deviation': df[col].std(),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurt()
        })
        stats.append(numerical_stats)

# Convert to DataFrame
report = pd.DataFrame(stats)

# Outlier Identification :
outlier_label = []
for col in report['Feature']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    LW = Q1 - 1.5 * IQR   # LW : Lower Whisker Line
    UW = Q3 + 1.5 * IQR   # UW : Upper Whisker Line
    outliers = df[(df[col] < LW) | (df[col] > UW)]
    if not outliers.empty:
        outlier_label.append("Has Outliers")
    else:
        outlier_label.append("No Outliers")

report["Outlier Comment"] = outlier_label

# Checking Report
#report.style.background_gradient(subset= ["Minimum", "Maximum", "Mean", "median", "Mode", "25%", "75%", "IQR", "Standard Deviation", "Skewness", "Kurtosis"], cmap= "coolwarm")

# Replace Outliers with Median Statergy

for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_count = outliers.sum()

    if outlier_count > 0:
        replacement = df[col].median()  
        df.loc[outliers, col] = replacement
       # print(f"Replaced {outlier_count} outliers in '{col}' with median.")
    # else:
    #     print(f"No outliers found in '{col}'.")


le = LabelEncoder()

df["Location_type"]  = le.fit_transform(df["Location_type"])
df["WH_capacity_size"] = le.fit_transform(df["WH_capacity_size"])
df["zone"] = le.fit_transform(df["zone"])
df["wh_owner_type"] = le.fit_transform(df["wh_owner_type"])

# Using PCA Concept:

# Step 1: Standardize the data

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# Step 2: Determine number of components to retain 90% variance

for i in range(1, df.shape[1] + 1):
    pca = PCA(n_components=i)
    pca.fit(X_scaled)
    evr = np.cumsum(pca.explained_variance_ratio_)
    if evr[i - 1] >= 0.90:
        pcs = i
        break

# print("Explained Variance Ratio:", evr)
# print("Number of components selected:", pcs)

# Step 3: Apply PCA

pca = PCA(n_components=pcs)
pca_data = pca.fit_transform(X_scaled)

# Step 4: Create DataFrame

pca_columns = [f'PC{j+1}' for j in range(pcs)]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)

# Step 5: Join Target Column with PCA:

pca_df = pca_df.join(df['product_wg_ton'], how = 'left')

#pca_df

from sklearn.model_selection import train_test_split

X = pca_df.drop(["product_wg_ton"], axis=1)
y = pca_df["product_wg_ton"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

RF = RandomForestRegressor()
RF.fit(X_train, y_train)

y_pred_RF = RF.predict(X_test)

r2_score_RF = r2_score(y_test, y_pred_RF)

print(f'The R2 Score for Random Forest :- {round(r2_score_RF * 100)}%') 