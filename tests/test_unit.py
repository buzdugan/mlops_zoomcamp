import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append("src")
import utils


def test_check_column_names():

    config = utils.load_config(file_path="config.yaml")
    file_path = Path(config['modelling_data_path'])
    target = config['target']

    df_quick_train = utils.read_dataframe(file_path, target, True)
    df_slow_train = utils.read_dataframe(file_path, target, False)

    actual_cols_quick_train = df_quick_train.columns.tolist()
    actual_cols_slow_train = df_slow_train.columns.tolist()

    expected_cols_quick_train = [
       'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
       'airbags', 'displacement', 'cylinder', 'turning_radius', 'length',
       'width', 'gross_weight', 'ncap_rating', 'region_code', 'segment',
       'model', 'fuel_type', 'max_torque', 'max_power', 'engine_type',
       'rear_brakes_type', 'transmission_type', 'steering_type',
       'claim_status'
    ]

    is_cols = [
        'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 
        'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper', 
        'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 
        'is_power_door_locks', 'is_central_locking', 'is_power_steering', 
        'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror', 
        'is_ecw', 'is_speed_alert'
    ]
    expected_cols_slow_train = expected_cols_quick_train + is_cols

    
    # Check the column names in the DataFrame

    assert set(actual_cols_quick_train) == set(expected_cols_quick_train)
    assert set(actual_cols_slow_train) == set(expected_cols_slow_train)
    
    # Check if "is_speed_alert" column is numeric
    assert pd.api.types.is_numeric_dtype(df_slow_train['is_speed_alert'])

    print("Column check test passed!")



if __name__ == "__main__":
    # Run the test when executing this script directly
    pytest.main([__file__])
