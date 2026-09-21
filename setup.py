from glob import glob
from setuptools import find_packages, setup


package_name = 'multicam_ipm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Only portable templates are installed. Local calibrations stay local.
        ('share/' + package_name + '/config', [
            'config/default.yaml',
            'config/surround.yaml',
        ]),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mduc',
    maintainer_email='mduc@todo.todo',
    description='Multi-camera inverse perspective mapping from a Kalibr camchain.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ipm_node = multicam_ipm.node:main',
            'surround_ipm_node = multicam_ipm.node:main',
        ],
    },
)
