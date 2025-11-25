import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Predictor class encapsulating feature prep & prediction
class AdvancedInjuryPredictor:
    def __init__(self, model, feature_names, target_classes):
        self.model = model
        self.feature_names = feature_names
        self.target_classes = target_classes
        self.le_player_type = LabelEncoder()
        self.le_player_type.fit(['Batsman', 'Bowler', 'All-Rounder'])

    def prepare_features(self, player_data):
        input_df = pd.DataFrame([player_data])

        # Example feature engineering
        input_df['workload_ratio'] = input_df['training_hours_week'] / (input_df['matches_last_month'] + 1)
        input_df['recovery_efficiency'] = input_df['previous_injuries_count'] / (input_df['recovery_days_last_injury'] + 1)

        # Default values for missing features
        default_features = {
            'bmi': 23.0,
            'body_fat_percent': 15.0,
            'bowling_overs_week': 10,
            'experience_years': max(1, player_data['age'] - 18),
            'stress_level': 5
        }
        for feature, default in default_features.items():
            if feature not in input_df.columns:
                input_df[feature] = default

        # Encode player type
        input_df['player_type_encoded'] = self.le_player_type.transform([player_data['player_type']])[0]

        # Ensure all features in the expected list are present
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
            return {
                'risk_level': self.target_classes[prediction],
                'confidence': confidence,
                'probabilities': dict(zip(self.target_classes, probabilities)),
                'recommendations': self.get_recommendations(self.target_classes[prediction])
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

# Streamlit app layout and logic
st.set_page_config(page_title="Cricket Injury Predictor", layout="wide")
st.title("Cricket Player Injury Risk Predictor")
st.write("Machine Learning system using Random Forest algorithm")

# Load model and feature info
try:
    model = joblib.load('injury_risk_model.pkl')  # Your saved model file
    feature_info = joblib.load('model_feature_info.pkl')  # Dict with 'feature_names' and 'target_classes'

    predictor = AdvancedInjuryPredictor(model, feature_info['feature_names'], feature_info['target_classes'])
    st.success("✅ ML Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Collect player data inputs
st.header("Player Assessment")

col1, col2 = st.columns(2)

with col1:
    player_type = st.selectbox("Player Type", ['Batsman', 'Bowler', 'All-Rounder'])
    age = st.slider("Age", 18, 40, 25)
    training_hours = st.slider("Training Hours/Week", 10, 40, 20)
    matches = st.slider("Matches Last Month", 1, 10, 4)

with col2:
    fatigue = st.slider("Fatigue Level (1-10)", 1, 10, 5)
    sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.5)
    injuries = st.slider("Previous Injuries Count", 0, 5, 1)
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
        'recovery_days_last_injury': 30  # Example fixed, can add input if you want
    }

    result = predictor.predict_with_confidence(player_data)

    if 'error' not in result:
        st.success("### Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", result['risk_level'])
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with col3:
            st.metric("ML Algorithm", "Random Forest")

        st.write("### Probability Distribution")
        prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Risk Level', 'Probability'])
        st.bar_chart(prob_df.set_index('Risk Level'))

        st.write("### Recommendations")
        for rec in result['recommendations']:
            st.write(f"• {rec}")
    else:
        st.error(f"Prediction failed: {result['error']}")

st.info("🔬 **Powered by Machine Learning**: Random Forest algorithm trained on balanced player data")

