# How to use Data Augmentation for HAABSA++

# Set up Environment

The source code is available at https://github.com/ofwallaart/HAABSA create a virtual environment as described.
  - Download c++ visual studio build tools (https://visualstudio.microsoft.com/downloads/). 
  - You need compatible drivers, Cuda (v9.0) and cuDNN (v6.4) (https://developer.nvidia.com/).

# Run Software

1. Make sure the environment is active ("Scripts\Activate.ps1")
2. Configure the main.py file
    - When using the supplied data, make sure only "runLCRROTALT_v4 = True", all others are "False".
4. In the environment, run `python main.py`

# How to obtain Augmented Data with BERT Word Embeddings

1. First run the ontology, to obtain the remaining test set
    - useOntology = True
2. Then augment the training data with:
    - loadData = True
    - augment_data = True
3. Save the acquired training data and remaining test data
4. Use getBERTusingColab.py in Google Colab to obtain the BERT word embeddings
5. Then use prepareBERT.py in the venv to get the desired data for HAABSA++.
