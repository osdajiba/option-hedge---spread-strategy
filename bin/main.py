#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: main.py

Description: This file launch the program, start core functions, loading data files, and running data analysis. Finally it outputs the results.

Author: Jacky

Date: 2024/04/19

Latest Edict Date: 2024/04/19

Version: 0.01

License: MIT

Usage: 
  - This script initializes log.
"""

from core import core


def main():
    print("Data Loading...")
    core.run()


if __name__ == '__main__':
    main()
