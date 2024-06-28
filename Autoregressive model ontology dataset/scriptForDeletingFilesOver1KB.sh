#!/bin/sh

# Define the size limit in bytes
SIZE_LIMIT=1024  # 1KB in bytes

# Loop over each directory in the current directory
for DIRECTORY in */ ; do
    # Ensure it's actually a directory
    if [ -d "$DIRECTORY" ]; then
        echo "Processing directory: $DIRECTORY"

        # Find all Proof*.json files in the directory
        find "$DIRECTORY" -name 'Proof*.json' -exec sh -c '
          for proof_file do
            # Extract the number from the filename
            i=$(echo "$proof_file" | sed "s/^.*Proof\([0-9]*\)\.json$/\1/")

            # Check if the file size exceeds the size limit
            if [ $(stat -c %s "$proof_file") -gt '$SIZE_LIMIT' ]; then
              echo "Deleting $proof_file"
              rm -v "$proof_file"

              # Define corresponding Input file based on the number
              input_file="${proof_file/Proof/Input}"

              # Delete the Input file if it exists, regardless of its size
              if [ -f "$input_file" ]; then
                echo "Deleting $input_file"
                rm -v "$input_file"
              fi
            fi
          done
        ' sh {} +
    fi
done

echo "Deletion complete."