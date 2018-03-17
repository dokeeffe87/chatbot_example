"""
Neural network chatbot.  Adapted from stanford tensorflow course.

"""

from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
from model import ChatBotModel
import config
import data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


