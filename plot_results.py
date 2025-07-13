import numpy as np
import pandas as pd
from scipy.special import ellipk, ellipe
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import csv


def plot_force(output):
  out_x = []
  out_y = []
  for i in range(len(output)):
    out_x.append(output[i][0])
    out_y.append(output[i][1])
  plt.plot(out_x, out_y)
  plt.grid(visible = True)
  plt.show()
  
def compute_derivative(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # Step size (distance between adjacent points)
    h = x[1] - x[0]

    # Initialize an array for the derivative
    derivative = np.zeros_like(y)

    # Apply finite difference method
    for i in range(1, len(x) - 1):
        derivative[i] = -(y[i+1] - y[i-1]) / (2 * h)  # Central difference

    # Handle the first and last points (forward and backward differences)
    derivative[0] = (y[1] - y[0]) / h  # Forward difference at the start
    derivative[-1] = (y[-1] - y[-2]) / h  # Backward difference at the end

    return derivative


def dynamic_func(D, f1, f2):
    return (f1 * D + f2 * D**3)

def calculate_r2_rmse(y_actual, y_predicted):
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)
    ss_residual = np.sum((y_actual - y_predicted)**2)
    return [1 - (ss_residual / ss_total),np.sqrt(np.mean((y_actual - y_predicted)**2))]

def calculate_rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted)**2))
  
def error_plot(output_df, numeric, ansys):
  error = numeric.iloc[:, 1] - ansys.iloc[:,1]
  error_pct = (error / numeric.iloc[:, 1]) * 100

  error_pct.iloc[len(error_pct)//2] = 0

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
  ax1.plot(output_df.iloc[:, 0], error, label='Error', marker='s')
  ax1.set_xlabel('origin_z [mm]')
  ax1.set_ylabel('Force_z [newton]')
  ax1.legend()
  ax1.set_title('Error between Numerical Model and ANSYS Simulation Output')
  ax1.grid(True)

  ax2.plot(output_df.iloc[:, 0], error_pct, label='Error Percentage', marker='d')
  ax2.set_xlabel('origin_z [mm]')
  ax2.set_ylabel('Error Percentage [%]')
  ax2.legend()
  ax2.set_title('Error Percentage')
  ax2.grid(True)

  plt.tight_layout()
  plt.show()

def plot_stiffness(disp, drv):
  plt.plot(disp, drv, color='r', linestyle='--')

  plt.grid(True)

  plt.show()