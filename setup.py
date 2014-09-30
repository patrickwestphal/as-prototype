
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A prototype for automatic, adaptive schema creation of RDF triples',
    'author': 'Patrick Westphal',
    'url': 'https://github.com/patrickwestphal/as-prototype',
    'download_url': 'https://github.com/patrickwestphal/as-prototype',
    'author_email': 'patrick.westphal@informatik.uni-leipzig.de',
    'version': '0.0.1',
    'install_requires': [
      'numpy==1.8.0',
      'scipy==0.12.1',
      'scikit-learn==0.14.1',
      'nose'
    ],
    'packages': ['proto'],
    'scripts': ['bin/runproto'],
    'name': 'as_prototype'
}

setup(**config)
