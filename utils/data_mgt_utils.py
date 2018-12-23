"""
Data management utils

Author: Joshua Owoyemi
Date: December 2018
"""
import re


"""
Key for sorting filenames according to serial number in the file names
Used as answer here: https://stackoverflow.com/questions/19366517/sorting-in-python-how-to-sort-a-list-containing-alphanumeric-values
"""
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]