#!/usr/bin/env python
# coding=utf-8

import unittest


if __name__ == "__main__":
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    discover = loader.discover('.', 'test_*.py')
    suite = unittest.TestSuite()
    suite.addTest(discover)
    runner.run(suite)
