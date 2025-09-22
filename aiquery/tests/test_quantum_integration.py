"""
tests/test_quantum_integration.py
Unit tests for quantum integration functionality.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_simulator import generate_quantum_synthetic_data, _simple_quantum_noise_sample
from calibration import clamp_dataframe, ks_compare
from bigquery_ai_integration import ingest_synthetic_to_bq

class TestQuantumSimulator(unittest.TestCase):
    """Test quantum simulator functionality"""
    
    def test_quantum_noise_sample(self):
        """Test quantum noise sample generation"""
        q_val = _simple_quantum_noise_sample(num_qubits=3, shots=64)
        self.assertIsInstance(q_val, float)
        self.assertGreaterEqual(q_val, 0.0)
        self.assertLessEqual(q_val, 1.0)
    
    def test_generate_quantum_synthetic_data_schema(self):
        """Test that generated data has correct schema"""
        df = generate_quantum_synthetic_data(num_samples=10, field_ids=2)
        
        # Check required columns
        expected_columns = [
            'timestamp', 'field_id', 'soil_ph', 'nitrogen', 'phosphorus', 
            'potassium', 'moisture', 'temperature', 'scenario', 'q_value'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertEqual(len(df), 10)
        # Accept datetime dtype or ISO8601 strings that can be parsed losslessly
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            # Validate parseability
            parsed = pd.to_datetime(df['timestamp'], utc=True, errors='raise')
            self.assertEqual(len(parsed), len(df))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['soil_ph']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['nitrogen']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['q_value']))
    
    def test_generate_quantum_synthetic_data_values(self):
        """Test that generated data has reasonable values"""
        df = generate_quantum_synthetic_data(num_samples=20, field_ids=3)
        
        # Check pH range
        self.assertTrue(df['soil_ph'].between(5.0, 8.0).all())
        
        # Check nutrient ranges
        self.assertTrue(df['nitrogen'].between(0.0, 50.0).all())
        self.assertTrue(df['phosphorus'].between(0.0, 30.0).all())
        self.assertTrue(df['potassium'].between(0.0, 50.0).all())
        
        # Check moisture range
        self.assertTrue(df['moisture'].between(0.0, 100.0).all())
        
        # Check temperature range
        self.assertTrue(df['temperature'].between(10.0, 35.0).all())
        
        # Check q_value range
        self.assertTrue(df['q_value'].between(0.0, 1.0).all())

    def test_clamp_dataframe(self):
        """Test clamping brings values into range"""
        df = generate_quantum_synthetic_data(num_samples=5, field_ids=2)
        df.loc[df.index[0], 'soil_ph'] = 20.0
        df2 = clamp_dataframe(df)
        self.assertTrue(df2['soil_ph'].between(3.5, 9.0).all())

    def test_ks_compare(self):
        """Test KS compare returns float and handles small samples"""
        df1 = generate_quantum_synthetic_data(num_samples=10, field_ids=2)
        df2 = generate_quantum_synthetic_data(num_samples=10, field_ids=2)
        p = ks_compare(df1['soil_ph'], df2['soil_ph'])
        self.assertIsInstance(p, float)
    
    def test_field_id_distribution(self):
        """Test that field IDs are distributed correctly"""
        df = generate_quantum_synthetic_data(num_samples=15, field_ids=3)
        
        # Check that we have the expected field IDs
        expected_fields = {'field_1', 'field_2', 'field_3'}
        actual_fields = set(df['field_id'].unique())
        self.assertEqual(actual_fields, expected_fields)
    
    def test_reproducibility(self):
        """Test that data generation is reproducible with same seed"""
        df1 = generate_quantum_synthetic_data(num_samples=5, field_ids=2)
        df2 = generate_quantum_synthetic_data(num_samples=5, field_ids=2)
        
        # Should be identical due to fixed seed
        pd.testing.assert_frame_equal(df1, df2)

class TestBigQueryIntegration(unittest.TestCase):
    """Test BigQuery integration functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_df = generate_quantum_synthetic_data(num_samples=5, field_ids=2)
    
    def test_dataframe_structure(self):
        """Test that test dataframe has correct structure"""
        self.assertEqual(len(self.test_df), 5)
        self.assertIn('timestamp', self.test_df.columns)
        self.assertIn('field_id', self.test_df.columns)
    
    def test_bigquery_ingest_fallback(self):
        """Test BigQuery ingest fallback behavior"""
        # This test will likely fail in CI without GCP credentials
        # but should not crash the test suite
        try:
            result = ingest_synthetic_to_bq(
                self.test_df, 
                "test_dataset", 
                "test_table"
            )
            self.assertIsInstance(result, str)
        except Exception as e:
            # Expected to fail without GCP credentials
            self.assertIn("credentials", str(e).lower())

class TestIntegration(unittest.TestCase):
    """Test end-to-end integration"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline without BigQuery"""
        # Generate data
        df = generate_quantum_synthetic_data(num_samples=10, field_ids=2)
        
        # Verify data structure
        self.assertEqual(len(df), 10)
        self.assertIn('q_value', df.columns)
        
        # Test CSV fallback
        output_path = "outputs/test_quantum_synthetic.csv"
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify file contents
        loaded_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(df, loaded_df)
        
        # Clean up
        os.remove(output_path)

def run_tests():
    """Run all tests"""
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_tests()
