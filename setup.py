#!/usr/bin/env python3

import os
import platform
from distutils.core import setup


install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'ipykernel',
    'jupyter',
    'gym',
    'tensorflow; platform_system!="Darwin"',
    'tensorflow-gpu; platform_system!="Darwin"',
    'tensorflow-macos; platform_system=="Darwin"',
    'tensorflow-metal; platform_system=="Darwin"',
    'pywin32; platform_system=="Windows"'
]

setup(
    name = 'dql',
    version = '0.1.0',
    description = 'Deep Q-Learning',
    author = 'Josef Hamelink',
    license = 'MIT',
    packages = ['dql'],
    install_requires = install_requires,
    entry_points = {
        'console_scripts': [
            'dql = dql.main:main'
        ]
    },
    data_files = [
        ('data' + os.sep + 'models', []),
        ('data' + os.sep + 'arrays', []),
        ('plots', [])
    ]
)
