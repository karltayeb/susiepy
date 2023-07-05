from setuptools import setup

setup(name='susiepy',
      version='0.1',
      description='Python/JAX implimentation of SuSiE/SuSiE adjacent',
      url='http://github.com/karltayeb/susiepy',
      author='Karl Tayeb',
      author_email='ktayeb@uchicago.edu',
      license='MIT',
      packages=['susiepy'],
      install_requires=[
            'jax',
            'jaxlib',
            'jaxopt',
            'numpy'
      ],
      zip_safe=False)