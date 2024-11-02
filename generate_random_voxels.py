import numpy as np
import pandas as pd

# Parameters for the simulation
num_subjects = 16    # Number of subjects
num_rois = 14752       # Number of regions of interest (ROI)
num_observations = 1  # Number of observations per subject-ROI combination

# Fixed effect for x
beta_0 = 2.0             # Overall intercept
beta_1 = 0.5             # Fixed effect of x (effect size for dummy variable)

# Random effect variances
subject_intercept_sd = 0.5  # Random intercept variance for subjects
roi_intercept_sd = 0.3      # Random intercept variance for ROI
roi_slope_sd = 0.2          # Random slope variance for ROI on x
residual_sd = 1.0           # Residual (error) standard deviation

# Simulate data
data = []

# Assign a consistent binary value for x to each subject (e.g., 0 or 1 for "gender")
subject_x = np.random.choice([0, 1], num_subjects)  # Randomly assign 0 or 1 to each subject

# Generate random effects for subjects and ROIs
subject_intercepts = np.random.normal(0, subject_intercept_sd, num_subjects)
roi_intercepts = np.random.normal(0, roi_intercept_sd, num_rois)
roi_slopes = np.random.normal(0, roi_slope_sd, num_rois)

# Simulate the observations
for subject in range(num_subjects):
    x = subject_x[subject]  # Get the fixed value of x for this subject
    subject_effect = subject_intercepts[subject]

    for roi in range(num_rois):
        roi_intercept = roi_intercepts[roi]
        roi_slope = roi_slopes[roi]

        # Generate multiple observations per subject-ROI pair
        for _ in range(num_observations):
            # Calculate y using the mixed-effects model structure
            y = (beta_0 + beta_1 * x +
                 subject_effect +       # Random intercept for subject
                 roi_intercept +        # Random intercept for ROI
                 roi_slope * x +        # Random slope for ROI on x
                 np.random.normal(0, residual_sd))  # Residual noise

            data.append([subject, roi, x, y])

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=["subject", "roi", "x", "y"])

#write to csv
df.to_csv(r'df2.csv', sep=',', index=False)