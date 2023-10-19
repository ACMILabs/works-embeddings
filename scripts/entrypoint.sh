#!/bin/ash

# Start with a fresh database
if [ "$DELETE_DB" = "true" ]; then
    rm -rf /code/works_db/
fi

# Check if a Chroma database exists on the filesystem
if [ ! -d "/code/works_db/" ] && [ "$REBUILD" = "false" ]; then
    echo "Directory /code/works_db/ does not exist. Downloading from ${CACHE_URL}..."

    if [ -n "$CACHE_URL" ]; then
        # Download file from S3
        cd /code/
        curl -O "${CACHE_URL}works_db.tar.gz"
        tar -xvzf works_db.tar.gz

        # Check for download success
        if [ $? -eq 0 ]; then
            echo "File downloaded successfully."
        else
            echo "Failed to download file."
            exit 1
        fi

    else
        echo "Please set CACHE_URL to download a cached database..."
    fi

else
    echo "Chroma database file /code/works_db/ exists..."
fi

# Start the web server
if [ "$DEBUG" = "true" ]; then
    echo "Starting Flask server..."
    python -u -m app.embeddings

else
    echo "Starting gunicorn server..."
    PYTHON_PATH=`python -c "import sys; print(sys.path[-1])"`
    gunicorn app.embeddings \
        --log-level DEBUG \
        --pythonpath $PYTHON_PATH \
        --workers 1 \
        --bind 0.0.0.0:80 \
        --timeout 120 \
        --reload
fi
