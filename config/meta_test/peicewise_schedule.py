_base_ = ["./meta_test_base.py"]



piecewise_schedule = dict(
    init_value=0.1,  
    boundaries_and_scales={60 * 8: 1.0, 120* 8: 0.1, 180* 8: 0.01} 
)