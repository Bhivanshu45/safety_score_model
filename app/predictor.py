"""
Model Predictor Service
Handles model loading and prediction logic
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import logging

from app.config import settings
from app.models import PredictionRequest, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Handles XGBoost model loading and predictions
    Ensures proper feature ordering and preprocessing
    """
    
    def __init__(self):
        """Initialize predictor with no model loaded"""
        self.model = None
        self.is_loaded = False
        self.feature_columns = settings.feature_columns
        self.threshold = settings.prediction_threshold
        
    def load_model(self, model_path: str = None) -> None:
        """
        Load the trained XGBoost model from disk
        
        Args:
            model_path: Path to the model .pkl file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if model_path is None:
            model_path = settings.model_file_path
            
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found at: {model_path}"
            )
        
        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
            # Log model info
            if hasattr(self.model, 'n_features_in_'):
                logger.info(f"Model expects {self.model.n_features_in_} features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")
    
    def _prepare_features(
        self, 
        input_data: Union[PredictionRequest, List[PredictionRequest]]
    ) -> pd.DataFrame:
        """
        Convert input data to properly formatted DataFrame
        Ensures correct feature order matching training data
        
        Args:
            input_data: Single or list of prediction requests
            
        Returns:
            DataFrame with features in correct order
        """
        # Convert single request to list
        if isinstance(input_data, PredictionRequest):
            input_data = [input_data]
        
        # Convert Pydantic models to dictionaries
        data_dicts = [req.model_dump() for req in input_data]
        
        # Create DataFrame
        df = pd.DataFrame(data_dicts)
        
        # Ensure feature order matches training
        df = df[self.feature_columns]
        
        # Ensure all columns are numeric (float64)
        for col in df.columns:
            df[col] = df[col].astype('float64')
        
        logger.info(f"Prepared {len(df)} samples with {len(df.columns)} features")
        logger.debug(f"Feature order: {list(df.columns)}")
        
        return df
    
    def predict(
        self, 
        input_data: Union[PredictionRequest, List[PredictionRequest]]
    ) -> Union[PredictionResponse, List[PredictionResponse]]:
        """
        Make prediction(s) using the loaded model
        
        Args:
            input_data: Single or list of prediction requests
            
        Returns:
            Single or list of prediction responses
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Track if input was single or batch
        is_single = isinstance(input_data, PredictionRequest)
        
        # Prepare features
        features_df = self._prepare_features(input_data)
        
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba(features_df)
            
            logger.info(f"Model probabilities shape: {probabilities.shape}")
            logger.info(f"First prediction probabilities: {probabilities[0] if len(probabilities) > 0 else 'None'}")
            
            # Extract probability of unsafe class (class 1)
            unsafe_probs = probabilities[:, 1]
            
            # Apply threshold for binary prediction
            binary_predictions = (unsafe_probs > self.threshold).astype(int)
            
            # Create response objects
            responses = []
            for prob, pred in zip(unsafe_probs, binary_predictions):
                response = PredictionResponse(
                    unsafe_probability=round(float(prob), 4),
                    prediction=int(pred)
                )
                responses.append(response)
            
            logger.info(
                f"Predictions made: {len(responses)} samples, "
                f"Unsafe count: {sum(binary_predictions)}"
            )
            
            # Return single response if input was single
            return responses[0] if is_single else responses
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction error: {str(e)}")
    
    def predict_batch(
        self, 
        input_data: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        """
        Batch prediction for multiple inputs
        
        Args:
            input_data: List of prediction requests
            
        Returns:
            List of prediction responses
        """
        return self.predict(input_data)
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update prediction threshold
        
        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        logger.info(f"Prediction threshold updated to: {threshold}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "threshold": self.threshold,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'n_features_in_'):
            info["model_n_features"] = self.model.n_features_in_
        
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        
        return info
    
    def predict_from_location(
        self, 
        latitude: float, 
        longitude: float, 
        dt: 'datetime',
        grid_mapper: 'GridMapper',
        use_default_lags: bool = True
    ) -> PredictionResponse:
        """
        Make prediction from location and datetime
        Converts lat/long to grid ID and datetime to temporal features
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            dt: datetime object
            grid_mapper: GridMapper instance for spatial conversion
            use_default_lags: If True, use default values (0.0) for lag features
            
        Returns:
            PredictionResponse with prediction result
            
        Raises:
            ValueError: If coordinates don't map to a grid
        """
        from app.datetime_utils import extract_temporal_features
        
        # Get grid ID from coordinates
        grid_id = grid_mapper.get_grid_id(latitude, longitude)
        
        if grid_id is None:
            raise ValueError(
                f"Coordinates ({latitude}, {longitude}) do not fall within any grid. "
                f"Please ensure coordinates are within the service area."
            )
        
        # Extract temporal features
        temporal_features = extract_temporal_features(dt)
        
        # IMPORTANT: Adjust year to training data range (2000-2020)
        # The model doesn't generalize well to years far outside training range
        if temporal_features['year'] > 2020:
            logger.info(f"Year {temporal_features['year']} is outside training range, using 2020")
            temporal_features['year'] = 2020
        elif temporal_features['year'] < 2000:
            logger.info(f"Year {temporal_features['year']} is outside training range, using 2000")  
            temporal_features['year'] = 2000
        
        # Create full feature dictionary
        # IMPORTANT: Lag features are critical for model predictions
        # Using realistic baseline values instead of 0.0
        # In production, these would come from a database
        if use_default_lags:
            # Use typical/average crime activity values
            # These are baseline values that assume moderate historical crime
            lag_features = {
                'lag_1': 1.5,          # Typical crime count in previous 4-hour window
                'lag_6': 1.8,          # Typical crime count 24 hours ago
                'lag_42': 1.6,         # Typical crime count 7 days ago
                'rolling_6_mean': 1.7, # Average over 24 hours
                'rolling_42_mean': 1.8 # Average over 7 days
            }
        else:
            # TODO: Fetch actual values from database based on grid_id and datetime
            lag_features = {
                'lag_1': 1.5,
                'lag_6': 1.8,
                'lag_42': 1.6,
                'rolling_6_mean': 1.7,
                'rolling_42_mean': 1.8
            }
        
        # Combine all features
        full_features = {
            'index_right': grid_id,
            **temporal_features,
            **lag_features
        }
        
        logger.info(f"Converted location ({latitude}, {longitude}) to grid {grid_id}")
        logger.info(f"Full features: {full_features}")
        
        # Create PredictionRequest and make prediction
        prediction_request = PredictionRequest(**full_features)
        result = self.predict(prediction_request)
        
        logger.info(f"Prediction result: prob={result.unsafe_probability}, pred={result.prediction}")
        
        return result
    
    def predict_route(
        self,
        coordinates: List[tuple],  # List of (lat, lng) tuples
        dt: 'datetime',
        grid_mapper: 'GridMapper'
    ) -> dict:
        """
        Predict safety score for a complete route (path from source to destination)
        
        Pipeline:
        1. Convert all (lat, lng) → grid IDs
        2. Extract temporal features (same for all points)
        3. Prepare lag features
        4. Batch predict all locations
        5. Aggregate using average and worst-case methods
        6. Return structured response
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            dt: datetime object for the route
            grid_mapper: GridMapper instance
            
        Returns:
            Dictionary with route safety metrics
            
        Raises:
            ValueError: If any coordinate doesn't map to a grid
            RuntimeError: If model not loaded
        """
        from app.datetime_utils import extract_temporal_features
        
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if len(coordinates) < 2:
            raise ValueError("Route must have at least 2 coordinates")
        
        logger.info(f"Starting route prediction for {len(coordinates)} coordinates")
        
        # =====================
        # STEP 1: Convert coordinates to grid IDs
        # =====================
        grid_ids = []
        for idx, (lat, lng) in enumerate(coordinates):
            grid_id = grid_mapper.get_grid_id(lat, lng)
            if grid_id is None:
                raise ValueError(
                    f"Coordinate {idx} ({lat}, {lng}) does not fall within any grid"
                )
            grid_ids.append(grid_id)
        
        logger.info(f"Converted all coordinates to grid IDs: {grid_ids}")
        
        # =====================
        # STEP 2: Extract temporal features (same for all points)
        # =====================
        temporal_features = extract_temporal_features(dt)
        
        # Adjust year to training range
        if temporal_features['year'] > 2020:
            logger.info(f"Year {temporal_features['year']} outside range, using 2020")
            temporal_features['year'] = 2020
        elif temporal_features['year'] < 2000:
            logger.info(f"Year {temporal_features['year']} outside range, using 2000")
            temporal_features['year'] = 2000
        
        logger.info(f"Temporal features: {temporal_features}")
        
        # =====================
        # STEP 3: Prepare lag features (defaults for route prediction)
        # =====================
        lag_features = {
            'lag_1': 1.5,
            'lag_6': 1.8,
            'lag_42': 1.6,
            'rolling_6_mean': 1.7,
            'rolling_42_mean': 1.8
        }
        
        # =====================
        # STEP 4: Create prediction requests for all coordinates
        # =====================
        prediction_requests = []
        for grid_id in grid_ids:
            full_features = {
                'index_right': grid_id,
                **temporal_features,
                **lag_features
            }
            pred_request = PredictionRequest(**full_features)
            prediction_requests.append(pred_request)
        
        logger.info(f"Created {len(prediction_requests)} prediction requests")
        
        # =====================
        # STEP 5: Batch predict all locations
        # =====================
        predictions = self.predict(prediction_requests)
        
        unsafe_probs = [p.unsafe_probability for p in predictions]
        binary_preds = [p.prediction for p in predictions]
        
        logger.info(f"Predictions: {unsafe_probs}")
        logger.info(f"Binary predictions: {binary_preds}")
        
        # =====================
        # STEP 6: Aggregate results (both average and worst-case)
        # =====================
        average_prob = sum(unsafe_probs) / len(unsafe_probs)
        worst_prob = max(unsafe_probs)
        unsafe_count = sum(binary_preds)
        
        # Use average probability for route_safety_score
        route_safety_score = average_prob
        
        # Map score to safety level
        if route_safety_score < 0.33:
            safety_level = "Low Risk"
        elif route_safety_score < 0.67:
            safety_level = "Medium Risk"
        else:
            safety_level = "High Risk"
        
        logger.info(
            f"Route aggregation: avg={average_prob:.4f}, worst={worst_prob:.4f}, "
            f"unsafe_count={unsafe_count}/{len(unsafe_probs)}"
        )
        
        # =====================
        # STEP 7: Return structured response
        # =====================
        return {
            'route_safety_score': round(route_safety_score, 4),
            'route_safety_level': safety_level,
            'worst_probability': round(worst_prob, 4),
            'average_probability': round(average_prob, 4),
            'segment_count': len(coordinates),
            'unsafe_segments': unsafe_count
        }
    
    def predict_all_grids(
        self,
        grid_mapper: 'GridMapper',
        dt: 'datetime' = None
    ) -> dict:
        """
        Predict safety score for all grid cells
        
        Returns complete heatmap data with geometries and safety scores.
        
        Args:
            grid_mapper: GridMapper instance with loaded grids
            dt: datetime object (uses current time if None)
            
        Returns:
            Dictionary with timestamp, statistics, and grid data with geometries
        """
        from app.datetime_utils import extract_temporal_features
        from datetime import datetime as dt_class
        
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if dt is None:
            dt = dt_class.now()
        
        logger.info(f"Starting heatmap prediction for all {len(grid_mapper.shapes_data)} grids")
        
        # =====================
        # STEP 1: Extract temporal features (same for all)
        # =====================
        temporal_features = extract_temporal_features(dt)
        
        # Adjust year to training range
        if temporal_features['year'] > 2020:
            temporal_features['year'] = 2020
        elif temporal_features['year'] < 2000:
            temporal_features['year'] = 2000
        
        # =====================
        # STEP 2: Prepare lag features (defaults)
        # =====================
        lag_features = {
            'lag_1': 1.5,
            'lag_6': 1.8,
            'lag_42': 1.6,
            'rolling_6_mean': 1.7,
            'rolling_42_mean': 1.8
        }
        
        # =====================
        # STEP 3: Create prediction requests for all grids
        # =====================
        grid_ids = []
        prediction_requests = []
        
        for geom, grid_id in grid_mapper.shapes_data:
            grid_ids.append(int(grid_id))
            
            full_features = {
                'index_right': int(grid_id),
                **temporal_features,
                **lag_features
            }
            pred_request = PredictionRequest(**full_features)
            prediction_requests.append(pred_request)
        
        logger.info(f"Created {len(prediction_requests)} prediction requests")
        
        # =====================
        # STEP 4: Batch predict all grids
        # =====================
        predictions = self.predict(prediction_requests)
        
        unsafe_probs = [p.unsafe_probability for p in predictions]
        binary_preds = [p.prediction for p in predictions]
        
        logger.info(f"Batch predictions complete. Probabilities: min={min(unsafe_probs):.4f}, max={max(unsafe_probs):.4f}")
        
        # =====================
        # STEP 5: Get grid geometries
        # =====================
        grids_geojson = grid_mapper.get_all_grids_geometries()
        
        # =====================
        # STEP 6: Build grid data with colors and statistics
        # =====================
        grid_data_list = []
        safe_count = 0
        low_risk_count = 0
        medium_risk_count = 0
        high_risk_count = 0
        avg_unsafe_probability = sum(unsafe_probs) / len(unsafe_probs)
        
        for grid_id, geojson_geom, unsafe_prob, binary_pred in zip(
            grid_ids, [g[1] for g in grids_geojson], unsafe_probs, binary_preds
        ):
            # Determine safety level and color (4-tier system)
            if unsafe_prob == 0.0:
                safety_level = "Safe"
                color = "#00ff00"  # Green
                safe_count += 1
            elif unsafe_prob <= 0.4:
                safety_level = "Low Risk"
                color = "#90EE90"  # Light Green
                low_risk_count += 1
            elif unsafe_prob <= 0.7:
                safety_level = "Medium Risk"
                color = "#ffff00"  # Yellow
                medium_risk_count += 1
            else:
                safety_level = "High Risk"
                color = "#ff0000"  # Red
                high_risk_count += 1
            
            grid_data = {
                "grid_id": int(grid_id),
                "unsafe_probability": round(float(unsafe_prob), 4),
                "safety_level": safety_level,
                "color": color,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": geojson_geom["coordinates"]
                }
            }
            grid_data_list.append(grid_data)
        
        logger.info(
            f"Heatmap complete: Safe={safe_count}, Low Risk={low_risk_count}, "
            f"Medium={medium_risk_count}, High={high_risk_count}, "
            f"Avg Unsafe Probability={avg_unsafe_probability:.4f}"
        )
        
        # =====================
        # STEP 7: Return structured response
        # =====================
        return {
            "timestamp": dt.isoformat(),
            "statistics": {
                "total_grids": len(grid_ids),
                "safe_count": safe_count,
                "low_risk_count": low_risk_count,
                "medium_risk_count": medium_risk_count,
                "high_risk_count": high_risk_count,
                "average_unsafe_probability": round(avg_unsafe_probability, 4)
            },
            "grids": grid_data_list
        }


# Create global predictor instance
predictor = ModelPredictor()


# Create global predictor instance
predictor = ModelPredictor()
