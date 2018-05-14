#!/home/eudie/miniconda3/envs/weddingPhotoSegregation/bin/python
# -*- coding: utf-8 -*-
# Author: Eudie

"""
This is the class for structuring the segregator. I will be using unsupervised method to help me tag the images.
And then will use supervised method for segregation.

"""

import pandas as pd
import numpy as np
import os
import cv2


class Segregator:
    def __init__(self, dir):
        self.dir = dir

    def extract_faces(self, face_dir):
        """
        This function extract all the faces from the photos in the main directory
        :param face_dir: directory where we want to save the faces, which we will used in unsurervised then tagging.
        :return: all the faces in all photos
        """
        # TODO

        return 0


    def cluster_faces(self, face_dir, clust_dir, no_of_clusters = 'Auto'):
        """
        It uses unsupervised method to cluster images. The purpose of this function is to help im manual tagging.
        By using this we will tagg the image which we will then used to classify
        :param face_dir: directory where all faces are stored
        :param clust_dir: directory where faces will be restored by cluster subfolders
        :param no_of_clusters: total number of cluster. Using 'Auto' algorithm will figure out itself.
        :return: clustered similar faces
        """

        # TODO

        return 0

    def train_model(self, tagged_face_dir, model_dir):
        """
        From this method we are going to traim the model on the tagged images after clustering and manuall tagging.
        The model will be later used for the final step of segregation.
        :param tagged_face_dir: face data directory which contain the subfolder with name of person,
        inside which the face image of her/his.
        :param model_dir: directory where you want to save the model
        :return: .ckpt file of the model
        """

        # TODO

        return 0

    def segregate(self, model_dir, output_dir):
        """
        Now we will use the trained model to segregate the images
        :param model_dir: folder where model is stored
        :param output_dir: folder where out segregate images should be stored
        :return: segregated images
        """

        # TODO

        return 0
