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
import pickle


class Segregator:
    """
    This module will be use to segregating the images.
    """
    def __init__(self, directory):
        self.directory = directory
        self.file_names = [os.path.join(root, filename)
                           for root, directories, filenames in os.walk(self.directory)
                           for filename in filenames
                           if filename.lower().endswith(('.jpg', '.jpeg'))]

    def extract_faces(self, scale_down=10, batch_size=32):
        """
        This function extract all the faces from the photos in the main directory
        :scale_down: factor by which images will be scaled down for processing
        """
        # TODO
        last_folder = os.path.basename(os.path.normpath(self.directory))
        self.face_dir = os.path.join(self.directory, '..', 'face_' + last_folder)
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)

        number_of_batches = len(self.file_names)//batch_size + 1
        for i in range(number_of_batches):
            end = min(len(self.file_names), (i+1)*batch_size)
            batch_images = []
            compressed_batch_images = []
            files = []

            for index, file in enumerate(self.file_names[i*batch_size:end]):
                print(file)
                image = cv2.imread(file)

                l, b, d = image.shape

                if index == 0:
                    batch_l, batch_b = min(l, b), max(l, b)

                if not (l == batch_l and b == batch_b):
                    new_l = batch_l
                    new_b = b * batch_l//l
                    dim = (new_b, new_l)
                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                    if new_b < batch_b:
                        padding_b = batch_b - new_b
                        padding = np.zeros((batch_l, padding_b, d), dtype=np.uint8)
                        image = np.concatenate((image, padding), axis=1)
                    else:
                        image = image[:, :batch_b, :]

                width = int(image.shape[1] / scale_down)
                height = int(image.shape[0] / scale_down)
                dim = (width, height)
                compressed_batch_images.append(cv2.resize(image, dim, interpolation=cv2.INTER_AREA))
                batch_images.append(image)
                files.append(file)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            batch_face_locations = face_recognition.batch_face_locations(compressed_batch_images)

            for j in range(len(batch_face_locations)):
                face_locations = batch_face_locations[j]
                image = batch_images[j]
                file = files[j]
                for index, face in enumerate(face_locations):
                    face = tuple(scale_down * x for x in face)
                    crop_face = image[face[0]:face[2], face[3]:face[1], :]
                    face_file_name = os.path.join(self.face_dir, "__".join((str(index),
                                                                            last_folder,
                                                                            os.path.split(file)[1])))
                    cv2.imwrite(face_file_name, crop_face)

    def cluster_faces(self, face_dir, encoding_pickle = None, no_of_clusters = 'Auto'):
        """
        It uses unsupervised method to cluster images. The purpose of this function is to help im manual tagging.
        By using this we will tagg the image which we will then used to classify
        :param face_dir: directory where all faces are stored
        :param clust_dir: directory where faces will be restored by cluster subfolders
        :param no_of_clusters: total number of cluster. Using 'Auto' algorithm will figure out itself.
        :return: clustered similar faces
        """

        # TODO

        if encoding_pickle is not None:
            with open(encoding_pickle, 'rb') as handle:
                encoding = pickle.load(handle)
        else:
            face_list = [os.path.join(root, filename)
                         for root, directories, filenames in os.walk(face_dir)
                         for filename in filenames
                         if filename.lower().endswith(('.jpg', '.jpeg'))]

            encoding = {}

            for face in face_list:
                face_image = cv2.imread(face)
                bbox = [(0, face_image.shape[1], face_image.shape[0], 0)]
                encoding[face] = face_recognition.face_encodings(face_image, known_face_locations=bbox, num_jitters=10)

            with open('face_encoding.pkl', 'wb') as handle:
                pickle.dump(encoding, handle)





        # self.cluster_dir = os.path.join(self.directory, 'cluster_faces')
        #
        # if not os.path.exists(self.cluster_dir):
        #     os.makedirs(self.cluster_dir)

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
