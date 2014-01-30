#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#Ivana Chingovska <ivana.chingovska@idiap.ch>
# Thu Feb  7 20:02:48 CET 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

import os, sys
import unittest
from .. import Quantization
import numpy

#############################################################################
# Tests blitz-based Quantization implementation
#############################################################################

class QuantizationTest(unittest.TestCase):

  def test01_Quantization(self):
    # Test quantization
    quant = Quantization('uint8')
    self.assertEqual(quant.num_levels, 256)
    self.assertEqual(quant.max_level, 255)
    self.assertEqual(quant.min_level, 0)
    self.assertEqual(quant.quantization_type, "uniform")
    self.assertEqual(quant.quantization_level(5), 5)
    
    quant = Quantization('uint16', "uniform", 8)
    self.assertEqual(quant.num_levels, 8)
    self.assertEqual(quant.max_level, 65535)
    self.assertEqual(quant.min_level, 0)
    img = numpy.array([8191, 8192, 16383, 16384], dtype='uint16')
    res = quant(img)
    self.assertTrue( (res == numpy.array([0,1,1,2])).all() )
    
    
    quant = Quantization('uint8', "uniform", 4, 64, 192)
    self.assertEqual(quant.num_levels, 4)
    self.assertEqual(quant.max_level, 192)
    self.assertEqual(quant.min_level, 64)
    self.assertTrue( (quant.quantization_table == numpy.array([64, 96, 128, 160])).all()  )
    img = numpy.array([[60, 95, 96], [127, 128, 129], [159, 160, 193]], dtype='uint8')
    res = quant(img)
    expected_res = numpy.array([[0, 0, 1], [1, 2, 2], [2, 3, 3]])
    self.assertTrue( (res == expected_res).all() )
    
    
    quantization_table = numpy.array([50, 100, 150, 200, 250], dtype='uint8')
    quant = Quantization('uint8', quantization_table = quantization_table)
    self.assertEqual(quant.num_levels, 5)
    self.assertEqual(quant.max_level, 255)
    self.assertEqual(quant.min_level, 50)
    img = numpy.array([0, 50, 99, 100, 101, 250, 255], dtype='uint8')
    res = quant(img)
    self.assertTrue( (res == numpy.array([0,0,0,1,1,4,4])).all() )

    quant = Quantization('uint8', 'uniform_rounding', 8)
    self.assertEqual(quant.num_levels, 8)
    self.assertEqual(quant.max_level, 255)
    self.assertEqual(quant.min_level, 0)
    self.assertTrue( (quant.quantization_table == numpy.array([0,19,55,91,127,163,199, 235])).all() )
   
   
