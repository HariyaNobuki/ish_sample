
def protected_div(x1, x2):
    if abs(x2) < 1e-8:
        return 1	# anything you like
    else:
        r = x1 / x2
        return r