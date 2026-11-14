from setuptools import setup
from glob import glob
import os

package_name = 'aruco_search'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniil Kavalevich',
    maintainer_email='daniil.kavalevich@gmail.com',
    description='ArUco cube detector and autonomous search action',
    license='MIT',
    entry_points={
        'console_scripts': [
            'aruco_detector = aruco_search.aruco_detector:main',
            'search_manager = aruco_search.search_manager:main',
        ],
    },
)
