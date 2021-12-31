from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="housing_value",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
