import numpy as np
import matplotlib.pyplot as plt

def compute_likelihood():
	theta_vals = np.linspace (0,1,101)	# theta = {0, 0.01, 0.02, ..., 1.0}
	l_vals = (theta_vals ** 5) * ((1 - theta_vals) ** 5)
	plt.plot(theta_vals, l_vals)
	plt.xlabel("theta")
	plt.ylabel("likelihood")
	plt.title("n = 10")
	plt.show()

compute_likelihood()