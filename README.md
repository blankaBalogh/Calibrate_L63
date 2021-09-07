# How to calibrate a neural network physical calibration ?
Supporting code & dataset - *submitted to Geophysical Research Letters.*

## Abstract
In current climate models, parameterization for each modeled process is tuned offline, individually. Thus, interactions between subgrid scale processes can be missed. To address this issue, a neural network (NN)-based emulator of the resulting parameterization is trained to predict subgrid-scale tendencies as a function of atmospheric state variables and some of the tuned model parameters, `theta`. Then, the fitted NN is implemented to replace the parameterization tuned offline. The optimal value of `theta` is determined by tuning its value online, with respect to a metric computing longterm prediction errors. 

Our approach has been demonstrated using the Lorenz’63 toy model (L63). The online optimization led to parameter values used to generate a reference dataset. In a second experiment, one of the model parameters was willingly biased. The resulting longterm prediction error was significantly reduced by optimizing online the value of one of `theta` parameters. 

## Code information
The code shared in the repository allows the replication of our experiments on L63 model.
