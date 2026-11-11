from pathlib import Path

from setuptools import find_packages, setup


root = Path(__file__).resolve().parent
runtime = [
    line.strip()
    for line in (root / 'requirements' / 'runtime.txt').read_text(
        encoding='utf-8').splitlines()
    if line.strip() and not line.startswith('#')
]

setup(
    name='psbg',
    version='0.1.0',
    description='Full PSBG model for infrared gas video segmentation',
    long_description=(root / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['mmseg', 'mmseg.*']),
    python_requires='>=3.8',
    install_requires=runtime,
    license='NVIDIA Source Code License for SegFormer',
    zip_safe=False)
