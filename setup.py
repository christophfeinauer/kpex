from setuptools import setup

setup(name='kpex',
      version='0.1',
      description='Keyphrase extraction with Python',
      url='https://github.com/christophfeinauer/kpex',
      author='Christoph Feinauer',
      author_email='christophfeinauer@gmail.com',
      include_package_data=True,
      package_data = {'': ['*.txt'],
                      'kpex': ['/kpex/tests/*txt']},
      license='',
      packages=['kpex'],
      zip_safe=False)
