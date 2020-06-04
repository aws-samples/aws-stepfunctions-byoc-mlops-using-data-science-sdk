# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io 
import sys
import signal
import traceback
from helper import *
import flask, pickle

import pandas as pd
NUM_CLASSES=2
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')#mask_rcnn_model_saved

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded


    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            num_classes = NUM_CLASSES
            cls.model = get_model_instance_segmentation(num_classes)
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            cls.model.load_state_dict(torch.load(os.path.join(model_path,'mask_rcnn_model_saved'), \
                                                             map_location=device))#,\
        
        return cls.model


    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""

        clf = cls.get_model()

        clf.eval()

        return clf.forward([input])

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    #health = ScoringService.get_model() is not None  # You can insert a health check here
    health = True

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if True:#flask.request.content_type == 'image/png':

        img = json.loads(flask.request.data.decode('utf-8'))
        img = np.asarray(img)

    else:
        return flask.Response(response='This predictor only supports png data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(img.shape))

    # Do the prediction
    trans = torchvision.transforms.ToTensor()
    img = np.array(img)
    img = img.astype(np.uint8)
    t1 = trans(img)
    
    
    

    #device = torch.device("cuda") 
    #t1 = t1.to(device)
    predictions = ScoringService.predict(t1)
    mask_response = predictions[0]['masks'].tolist()
    result = json.dumps(mask_response)


    return flask.Response(response=result, status=200, mimetype='text/csv')
