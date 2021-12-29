import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

short_description = """
    A convience package for running many types of agents through
    an OpenAI 'gym' environment. Includes 3 agents, several
    neural Q function approximators, and a grid search class.
"""

setuptools.setup(
    name="gym_runner",
    version="0.0.1",
    author="Alex Keeney",
    author_email="alex.keeney766@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexkeeney766/rl_with_function_approximation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "gym>=0.21.0",
        "tqdm>=4.62.3",
        "scikit-learn>=1.0.1",
    ],
)
