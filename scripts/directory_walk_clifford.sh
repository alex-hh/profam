#!/bin/bash

# Function to list contents of a directory
list_contents() {
    local dir="$1"
    echo "Contents of $dir:"
    ls -1 "$dir"
    echo ""
}

# Function to recursively walk through directories
walk_directories() {
    local current_dir="$1"

    # List contents of current directory
    list_contents "$current_dir"

    # Loop through items in the current directory
    for item in "$current_dir"/*; do
        if [ -d "$item" ]; then
            # If item is a directory, recursively walk through it
            walk_directories "$item"
        fi
    done
}

walk_directories "/state/partition1/ProFam_data"