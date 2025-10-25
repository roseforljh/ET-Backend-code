#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# This script can be expanded with additional startup logic,
# such as running database migrations or waiting for other services.

# Execute the main command passed as arguments to this script.
# In our Dockerfile, this will be `python run.py`.
exec "$@"