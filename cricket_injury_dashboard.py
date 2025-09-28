# cricket_injury_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Define the predictor class IN THE SAME FILE
class AdvancedInjuryPredictor:
    def __init__(self, model, feature_names, target_classes):
        self.model = model
        self.feature_names = feature_names
        self.target_classes = target_classes
        self.le_player_type = LabelEncoder()
        self.le_player_type.fit(['Batsman', 'Bowler', 'All-Rounder'])
        
    def prepare_features(self, player_data):
        input_df = pd.DataFrame([player_data])
        
        # Feature engineering
        input_df['workload_ratio'] = input_df['training_hours_week'] / (input_df['matches_last_month'] + 1)
        input_df['recovery_efficiency'] = input_df['previous_injuries_count'] / (input_df['recovery_days_last_injury'] + 1)
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
                
        return input_df
    
    def predict_with_confidence(self, player_data):
        try:
            input_df = self.prepare_features(player_data)
            X_new = input_df[self.feature_names]
            
            prediction = self.model.predict(X_new)[0]
            probabilities = self.model.predict_proba(X_new)[0]
            confidence = max(probabilities)
            
            # Risk score calculation
            risk_score = min(100, max(0, 
                player_data.get('fatigue_level', 5) * 3 +
                player_data.get('previous_injuries_count', 0) * 8 +
                min(player_data.get('training_hours_week', 20) * 0.5, 15)
            ))
            
            return {
                'risk_level': prediction,
                'confidence': confidence,
                'risk_score': risk_score,
                'probabilities': dict(zip(self.target_classes, probabilities)),
                'recommendations': self.get_recommendations(prediction)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_recommendations(self, risk_level):
        if risk_level == 'High':
            return [
                "Reduce training intensity by 40-50%",
                "Consult physiotherapist immediately",
                "Take 2-3 days complete rest"
            ]
        elif risk_level == 'Medium':
            return [
                "Reduce training by 20-30%",
                "Increase recovery time",
                "Focus on sleep quality"
            ]
        else:
            return [
                "Maintain current training",
                "Continue recovery practices",
                "Monitor fatigue levels"
            ]

# Streamlit app
st.set_page_config(page_title="Cricket Injury Predictor", layout="wide")
st.title("Cricket Player Injury Risk Predictor")

# Simple model loading
try:
    # Load the basic model components
    basic_model = joblib.load('best_injury_predictor.pkl')
    feature_info = joblib.load('model_features.pkl')
    
    # Create predictor
    predictor = AdvancedInjuryPredictor(
        basic_model.named_steps['classifier'],
        feature_info['feature_names'],
        feature_info['target_classes']
    )
    
    st.success("✅ Model loaded successfully!")
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Prediction interface
st.header("Player Assessment")

col1, col2 = st.columns(2)

with col1:
    player_type = st.selectbox("Player Type", ["Batsman", "Bowler", "All-Rounder"])
    age = st.slider("Age", 18, 40, 25)
    training_hours = st.slider("Training Hours/Week", 10, 40, 20)
    matches = st.slider("Matches/Month", 1, 10, 4)

with col2:
    fatigue = st.slider("Fatigue Level (1-10)", 1, 10, 5)
    sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.5)
    injuries = st.slider("Previous Injuries", 0, 10, 1)
    fitness = st.slider("Fitness Score", 40, 100, 75)

if st.button("Predict Injury Risk", type="primary"):
    player_data = {
        'player_type': player_type,
        'age': age,
        'training_hours_week': training_hours,
        'matches_last_month': matches,
        'fatigue_level': fatigue,
        'sleep_hours_day': sleep,
        'previous_injuries_count': injuries,
        'fitness_score': fitness,
        'experience_years': max(1, age-18),
        'recovery_days_last_injury': 30,
        'stress_level': 5
    }
    
    result = predictor.predict_with_confidence(player_data)
    
    if 'error' not in result:
        st.success("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", result['risk_level'])
        with col2:
            st.metric("Risk Score", f"{result['risk_score']}/100")
        with col3:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        st.write("### Recommendations")
        for rec in result['recommendations']:
            st.write(f"• {rec}")
    else:
        st.error(f"Prediction failed: {result['error']}")