_base_ = ["./meta_test_base.py"]



piecewise_schedule = dict(
    init_value=0.1,  
    boundaries_and_scales={5000: 0.1, 10000: 0.1},
)