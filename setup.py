from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'go1_mujoco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='siddarth',
    maintainer_email='siddarth.dayasagar@gmail.com',
    description='GO2 Mujoco Isaac Sim integration test node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_move = go1_mujoco.isaac_test:main',
        ],
    },
)
