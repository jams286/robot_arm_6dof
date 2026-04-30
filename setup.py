from setuptools import setup, find_packages

setup(
    name="robot-arm-6dof",
    version="1.0.0",
    description="6-DOF revolute serial manipulator simulator",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
)
