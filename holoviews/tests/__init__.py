try:
    # Standardize backend due to random inconsistencies
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
except ImportError:
    pass
