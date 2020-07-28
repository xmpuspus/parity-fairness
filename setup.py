import pathlib
from setuptools import setup

# from os import path
# this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

# The directory containing this script
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Call setup
setup(
  name = 'parity-fairness', # How you named your package folder (MyLib)
  packages = ['parity'], # Chose the same as "name"
  version = '0.1.29', # Start with a small number and increase it with every change you make
  license='MIT', # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Measure bias from data and machine learning models.', # Give a short description about your library
  long_description=README,
  long_description_content_type="text/markdown",
  author = 'Xavier M. Puspus', # Type in your name
  author_email = 'xpuspus@gmail.com', # Type in your E-Mail
  url = 'https://github.com/xmpuspus/parity-fairness', # Provide either the link to your github or to your website
  download_url = 'https://github.com/xmpuspus/parity-fairness/archive/v_01.29.tar.gz', # I explain this later on
  keywords = ['fairness', 'bias', 'explainability', 'AI', 'machine learning', 'data'], # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'matplotlib',
          'joblib',
          'numpy',
          'altair',
          'pandas',
          'scikit_learn',
          'sklearn', 
          'aif360',
          'plotly',
          'Ipython',
          'sklearn',
          'fairlearn',
          'dice_ml',
          'interpret',
          'BlackBoxAuditing',
          'witwidget',
          'boto3'],
    
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
 