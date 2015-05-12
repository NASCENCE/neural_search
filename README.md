# Neural Search

## Dependencies

  - brainstorm
  - h5py
  - ipython
  - numpy 
  - sacred

## Run
First train a network using `train_net.py`. 
Configuration can be adjusted form the commandline using the `with` statement. 
See sacred documentation for more info.

    > python train_net.py

This will train a network on the data and save the weights to `best_weights.npy`. 
This should take ca. 45sec per epoch and ca. 50 epochs. 

## Investigate Errors
After you've trained a network you can investigate its errors by running the `Investigate Errors.ipynb` notebook.
To do so first start an ipython notebook (might cause additional missing dependencies):

    > ipython notebook

Then open the file from your browser.

## Search for Configurations
After you've trained a network you can use it to search for configurations using the `Search.ipynb` notebook. (see above)

