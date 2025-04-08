import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate a large population (this is our ground truth)
# np.random.seed(42)
N = 100_000
mu = 5
sigma = 2
population = np.random.normal(loc=mu, scale=sigma, size=N)

# True population variance
true_variance = np.var(population)  # uses n by default in np.var()

# Step 2: Sample repeatedly from the population
n = 30        # sample size
repeats = 1000
sample_means = []
biased_vars = []
unbiased_vars = []

for _ in range(repeats):
    sample = np.random.choice(population, size=n, replace=False)
    mean_sample = np.mean(sample)
    sample_means.append(mean_sample)

    # Biased variance (dividing by n)
    biased = np.var(sample, ddof=0, mean=mu)
    
    # Unbiased variance (dividing by n - 1)
    unbiased = np.var(sample, ddof=1, mean=mu)
    
    biased_vars.append(biased)
    unbiased_vars.append(unbiased)
bins = int(np.sqrt(len(biased_vars)))
# Step 3: Plot the results
plt.figure(figsize=(10, 6))
plt.hist(biased_vars, bins=bins, alpha=0.5, label="Biased (ddof=0)")
plt.hist(unbiased_vars, bins=bins, alpha=0.5, label="Unbiased (ddof=1)")
plt.axvline(true_variance, color='red', linestyle='--', label="True Variance")
plt.title("Biased vs Unbiased Variance Estimation")
plt.xlabel("Estimated Variance")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Print average results
print(f"True population variance     : {true_variance:.4f}")
print(f"Average biased sample var    : {np.mean(biased_vars):.4f}")
print(f"Average unbiased sample var  : {np.mean(unbiased_vars):.4f}")
print(f"Average sample mean          : {np.mean(sample_means):.4f}")