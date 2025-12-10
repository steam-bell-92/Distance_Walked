import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

class HealthActivityAnalyzer:
    def __init__(self):
        """
        Initializes the analyzer with models and scaler.
        """
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.features_list = None # To ensure column alignment for future predictions
        
        # Define models with the parameters you specified
        self.models = {
            'Linear Regression': LinearRegression(),
            'SVR (Poly)': SVR(kernel='poly', C=3, epsilon=0.2, degree=3),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        
        # specific SVR visualization context
        # 
        
        self.results = {} # To store MAE, RMSE, R2 for comparison

    def load_data(self, filepath):
        """
        Loads data and handles the initial duplication cleanup.
        """
        self.data = pd.read_csv(filepath)
        initial_shape = self.data.shape
        
        # FIX: Reassigning to self.data to actually drop duplicates
        self.data = self.data.drop_duplicates()
        
        print(f"Data Loaded. Rows dropped: {initial_shape[0] - self.data.shape[0]}")
        print(f"Current Shape: {self.data.shape}")

    def feature_engineering(self):
        """
        Handles Missing values, BP parsing, and Unit conversions.
        """
        df = self.data.copy()
        
        # 1. Handle Missing Sleep Disorder
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
        
        # 2. Convert Steps to Distance
        # Note: 0.00074 is the average km per step
        df['Distance_walked(km)'] = df['Daily Steps'] * 0.00074
        
        # 3. Parse Blood Pressure
        # 

[Image of systolic vs diastolic blood pressure readings]

        # Splitting "120/80" into two numeric features
        df['BP_systolic'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
        df['BP_diastolic'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
        
        # 4. Drop redundant columns
        df.drop(['Blood Pressure', 'Daily Steps'], axis=1, inplace=True)
        
        # 5. One Hot Encoding
        cols_to_encode = ['Occupation', 'Sleep Disorder', 'BMI Category', 'Gender']
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        
        self.data = df
        print("Feature Engineering Complete.")

    def prepare_training_data(self):
        """
        Splits data and applies RobustScaling.
        """
        X = self.data.drop('Distance_walked(km)', axis=1)
        y = self.data['Distance_walked(km)']
        
        # Save feature columns for production safety
        self.features_list = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale Features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("Data Split and Scaled.")

    def train_and_evaluate(self):
        """
        Loops through all models, trains them, and stores metrics.
        """
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = root_mean_squared_error(self.y_test, y_pred)
            
            # Cross Validation (using training data)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10)
            avg_cv_r2 = cv_scores.mean()
            
            # Store results
            self.results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'CV R2': avg_cv_r2,
                'Prediction': y_pred # Storing for plotting later
            }
            
            print(f"  MAE: {mae:.4f}")
            print(f"  CV R2: {avg_cv_r2:.4f}")

    def plot_model_comparison(self):
        """
        Generates the bar chart comparing all models.
        """
        models = list(self.results.keys())
        mae = [self.results[m]['MAE'] for m in models]
        rmse = [self.results[m]['RMSE'] for m in models]
        r2 = [self.results[m]['CV R2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width, mae, width, label='MAE', color='skyblue')
        rects2 = ax.bar(x, rmse, width, label='RMSE', color='orange')
        rects3 = ax.bar(x + width, r2, width, label='CV R2', color='lightgreen')

        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_predictions(self, model_name='Linear Regression'):
        """
        Scatter plot of Actual vs Predicted for a specific model.
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found.")
            return

        y_pred = self.results[model_name]['Prediction']
        
        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.6, color='tomato')
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        plt.title(f'Actual vs Predicted: {model_name}')
        plt.xlabel('Actual Distance (km)')
        plt.ylabel('Predicted Distance (km)')
        plt.grid(True, alpha=0.5)
        plt.show()

# --- Usage ---
if __name__ == "__main__":
    # Initialize
    analyzer = HealthActivityAnalyzer()
    
    # Load Data (Replace with your actual path)
    # analyzer.load_data('Health_dataset.csv')
    
    # Using dummy data creation for demonstration if file missing
    try:
        analyzer.load_data('Health_dataset.csv')
    except FileNotFoundError:
        print("Dataset not found. Please upload 'Health_dataset.csv'")
    
    if analyzer.data is not None:
        # Pipeline execution
        analyzer.feature_engineering()
        analyzer.prepare_training_data()
        analyzer.train_and_evaluate()
        
        # Visualizations
        analyzer.plot_model_comparison()
        analyzer.plot_predictions('Random Forest')