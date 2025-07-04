from datetime import timedelta

def daterange(start_date, end_date):
    """
    Generator to yield dates from start_date to end_date inclusive.
    """
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)
