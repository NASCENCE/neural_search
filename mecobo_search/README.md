# Neural Search on Single Wall Carbon Nanotubes

Here is the code we used to train a neural network to model the SWCN material and search for logic gates.

## Data
The file `data.txt` contains roughly 54K samples measured from the MECOBO board.
It was preprocessed using the `PrepareData.ipynb` notebook which stores the result as `data50K.h5`.

## Neural Network
To train the neural network model we used just run: 
    
    > python trial.py with best_net

This will train a neural network and save it as `best_net.h5`.

## Investigate Model Quality
In the `InvestigateFit.ipynb` notebook we investigated how well the NN models the data.

## Search for Gates
Finally we use our Neural Network to search for the logic gates AND, OR, NAND, NOR, and XOR.
We then run the resulting configurations on the real hardware and compare the results.
All of this is done in the `Search.ipynb` notebook


