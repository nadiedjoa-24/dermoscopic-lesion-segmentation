import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.normpath(os.path.join(BASEDIR,'..','dataset','melanoma','ISIC_0000146 .jpg'))
SAVE_PATH = os.path.normpath(os.path.join(BASEDIR, 'test'))