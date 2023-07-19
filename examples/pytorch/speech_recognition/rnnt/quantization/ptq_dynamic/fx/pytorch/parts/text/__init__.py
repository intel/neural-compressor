# Copyright (c) 2017 Keith Ito
""" from https://github.com/keithito/tacotron """
from . import cleaners


def _clean_text(text, cleaner_names, *args):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text, *args)
    return text
