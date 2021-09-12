import setuptools

with open("README.md") as file:
    read_me_description = file.read()

setuptools.setup(
    name="mewtwo",
    version="1.0",
    author="Ahmed Aljeshi, Juan Francisco Balbi, Paula Escusol Entio, Isobel Rae Impas, Paliz Mungkaladung",
    author_email="ahmed@student.ie.edu, jfbalbi@student.ie.edu, paula.escusol@student.ie.edu, isobel.impas@student.ie.edu, paliz.mungkaladung@student.ie.edu",
    description="A class containing various methods useful for analyzing data and building a Risk Based Segmentation model",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaulaEscusol/Group-assignment_Python-II",
    packages = ['mewtwo'],
    install_requires=['pandas', 'numpy', 'scikit-learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)