from xml.etree.ElementInclude import include
import setuptools

setuptools.setup(
    name="cispa",
    version="0.0.1",
    author="Zhitao Gan",
    author_email="zgan9@jhu.edu",
    description="cispa",
    long_description="cispa",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "rich",
        "click",
        
    ],
    include_package_data=True,
    python_requires=">=3.9",
)
