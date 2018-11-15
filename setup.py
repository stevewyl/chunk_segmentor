from setuptools import setup, find_packages

with open('requirements.txt') as requirements:
    REQUIREMENTS = requirements.readlines()
long_description = open('README.md', encoding='utf-8').read()

REQUIREMENTS = ['seqeval>=0.0.3', 'Keras>=2.2.0',
                'tensorflow>=1.9.0', 'JPype1>=0.6.3',
                'numpy>=1.14.3', 'scikit-learn>=0.19.1',
                'hanziconv>=0.3.2']

setup(
    name='chunk_segmentor',
    version='1.1.0',
    description='Segmentor with Noun Pharses',
    long_description=long_description,
    author='yilei.wang',
    author_email='stevewyl@163.com',
    license='MIT',
    install_requires=REQUIREMENTS,
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/stevewyl/chunk_segmentor',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Java'
    ]
)

