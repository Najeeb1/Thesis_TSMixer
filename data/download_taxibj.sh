
#!/bin/bash

# URL of the Dropbox file
DROPBOX_URL="https://www.dropbox.com/sh/l9drnyeftcmy3j1/AACCgUyOj2akPNBwFAe9W1-ia?e=1&dl=0"

# Destination path to save the downloaded file
DESTINATION_PATH="taxibj/dataset.npz"

# Download the file using curl
curl -L -o "$DESTINATION_PATH" "$DROPBOX_URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "File downloaded successfully!"
else
    echo "Failed to download the file."
fi
