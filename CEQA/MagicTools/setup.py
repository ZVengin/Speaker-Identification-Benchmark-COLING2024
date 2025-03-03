from setuptools import setup

setup(
    name='MagicTools',
    version='0.1.0',
    description='A package for frequently used operations',
    url='git@github.com:ZVengin/MagicTools.git',
    author='ZVengin',
    author_email='wenjiezhong717@gmail.com',
    license='BSD 2-clause',
    packages=['MagicTools'],
    install_requires=['nltk','torch','tqdm','transformers','pandas','tensorboard'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
