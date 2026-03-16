from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aws-ple-platform")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
