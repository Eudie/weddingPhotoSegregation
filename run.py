#!/home/eudie/miniconda3/envs/weddingPhotoSegregation/bin/python
# -*- coding: utf-8 -*-
# Author: Eudie

"""
running the class

"""
import photoSegregator
data_folder = '/media/eudie/Seagate Backup Plus Drive/WEDDING-YOGITA-UTKARSH/1-PHOTOS/1- AK- SAGAI-SANGEET-RECEPTION-VIDISHA'
seg = photoSegregator.Segregator(data_folder)
seg.extract_faces(batch_size=5)
