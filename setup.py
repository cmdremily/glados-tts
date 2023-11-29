from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='glados_tts',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    # Other optional metadata
    author='Cmdr. Emily',
    author_email='cmdr@cmdr.com',
    description='A TTS package for GLaDOS.',
    url='https://github.com/cmdremily/glados-tts',
)