# SympleticNets

## Description
Goal: implementation of sympletic networks to predict ODE's solution in time.
Inputs are the positions on the X axis of the two objects and their velocity at time t
Outputs are their positions on the X-axis and their velocity at time t+1

## Content
| File name | Description |
| ----------- | ----------- |
| main.py | Main file |
| utils.py | Basic utility functions for the variables, plots...|
| networks.py | Networks implementation (both sympletic and non-sympletic) |

## Remarks
- Required packages: pandas, numpy, pytorch (pip install *package_name*)
- Changes to run on GPU: file name ('\' to '/')
- Delete all plots to run on GPU and use `DataFrame.to_csv()` to get the prediction values
