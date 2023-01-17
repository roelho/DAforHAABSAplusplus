# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow as tf
import cabascModel
import lcrModel
import lcrModelInverse
import lcrModelAlt
import svmModel
from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys

import lcrModelAlt_hierarchical_v4

# main function
def main(_):
    loadData         = False         # only for non-contextualised word embeddings.
                                        # Use prepareBERT for BERT (and BERT_Large) and prepareELMo for ELMo
    augment_data = False              # Load data must be true to augment
    useOntology      = False            # When run together with runLCRROTALT, the two-step method is used
    runLCRROTALT     = False

    runSVM           = False
    runCABASC        = False
    runLCRROT        = False
    runLCRROTINVERSE = False
    weightanalysis   = False

    runLCRROTALT_v4     = True


    #determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM or runLCRROTALT_v4:
        backup = True
    else:
        backup = False
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector , ct = loadDataAndEmbeddings(FLAGS, loadData, augment_data)
    print(test_size)
    remaining_size = 248
    accuracyOnt = 0.868

    if useOntology == True:
        print('Starting Ontology Reasoner')
        #in sample accuracy
        Ontology = OntReasoner()
        accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path_ont, runSVM)
        #out of sample accuracy
        #Ontology = OntReasoner()      
        #accuracyInSampleOnt,   InSample_size = Ontology.run(backup,FLAGS.train_path_ont, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
            print(test[0])
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    # LCR-Rot-hop model
    if runLCRROTALT == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size, augment_data, FLAGS.augmentation_file_path, ct)
       tf.reset_default_graph()

    if runLCRROTALT_v4 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path,
                                                                       test,
                                                                       accuracyOnt,
                                                                       test_size,
                                                                       remaining_size,
                                                                       augment_data,
                                                                       FLAGS.augmentation_file_path,
                                                                       ct)
       tf.reset_default_graph()

print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
