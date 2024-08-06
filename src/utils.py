import pendulum

def get_current_date():
    return pendulum.now().strftime('%Y%m%d_%H%m%S')