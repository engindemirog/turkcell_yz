from setuptools import setup, find_packages

setup(
    name="ecommerce_model",
    version="1.0.0",
    description="E-ticaret sipariş iptal tahmini için FastAPI servisi",
    author="Engin Demiroğ",
    packages=find_packages(include=["", ".*"]),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
    ],
    entry_points={
        "console_scripts": [
            "train_model = train",  
            "run-ecommerce-api=main:app"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Framework :: FastAPI",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)