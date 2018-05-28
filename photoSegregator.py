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
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Segregator:
    """
    This module will be use to segregating the images.
    """
    def __init__(self, directory):
        self.directory = directory
        self.file_names = [os.path.join(root, filename)
                           for root, directories, filenames in os.walk(self.directory)
                           for filename in filenames
                           if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def extract_faces(self, scale_down=10):
        """
        This function extract all the faces from the photos in the main directory
        :scale_down: factor by which images will be scaled down for processing
        """
        # TODO
        self.face_dir = os.path.join(self.directory, '..', 'face_' + os.path.basename(os.path.normpath(self.directory)))
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)

        for file in self.file_names:
            image = cv2.imread(file)
            width = int(image.shape[1] / scale_down)
            height = int(image.shape[0] / scale_down)
            dim = (width, height)
            resize_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(resize_image, model="cnn")
            for index, face in enumerate(face_locations):
                face = tuple(scale_down * x for x in face)
                crop_face = image[face[0]:face[2], face[3]:face[1], :]
                face_file_name = os.path.join(self.face_dir, "__".join((str(index), os.path.split(file)[1])))
                cv2.imwrite(face_file_name, crop_face)


    def cluster_faces(self, clust_dir, no_of_clusters = 'Auto'):
        """
        It uses unsupervised method to cluster images. The purpose of this function is to help im manual tagging.
        By using this we will tagg the image which we will then used to classify
        :param face_dir: directory where all faces are stored
        :param clust_dir: directory where faces will be restored by cluster subfolders
        :param no_of_clusters: total number of cluster. Using 'Auto' algorithm will figure out itself.
        :return: clustered similar faces
        """

        # TODO
        self.cluster_dir = os.path.join(self.directory, 'cluster_faces')

        if not os.path.exists(self.cluster_dir):
            os.makedirs(self.cluster_dir)

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
