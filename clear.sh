# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <extension>"
    exit 1
fi

EXTENSION=$1
find reconstructed/ . -name "*.${EXTENSION}" -type f -exec rm {} +