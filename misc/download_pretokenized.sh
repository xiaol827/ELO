# # --- (8) Download 24B data ---
# Configure R2 access
source .venv/bin/activate
export DATA_ROOT="$PWD/data/dclm_tokenized"
mkdir -p "$DATA_ROOT"

export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
ENDPOINT="https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
pip install awscli

echo "Downloading first 23 shards in parallel..."

# Function to download a single shard with retry logic
download_shard() {
    local shard_num=$1
    local shard_file=$(printf "train_%06d.npy" $shard_num)
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        echo "Downloading $shard_file (attempt $((retry_count + 1))/$max_retries)..."
        if aws s3 --endpoint-url "$ENDPOINT" cp "s3://pretokenized-dataset/$shard_file" "$DATA_ROOT/$shard_file" --no-progress; then
            echo "✓ Successfully downloaded $shard_file"
            return 0
        else
            echo "✗ Failed to download $shard_file (attempt $((retry_count + 1)))"
            retry_count=$((retry_count + 1))
            sleep 2
        fi
    done

    echo "✗ Failed to download $shard_file after $max_retries attempts"
    return 1
}

# Export the function so it can be used by xargs
export -f download_shard
export DATA_ROOT ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

# Generate list of shard numbers (0-22) and download in parallel
# Use -P 8 for 8 parallel downloads (adjust based on your bandwidth/system)
seq 0 59 | xargs -n 1 -P 8 -I {} bash -c 'download_shard {}'

# Check if all downloads completed successfully
echo "Verifying downloaded shards..."
missing_files=0
for i in $(seq 0 59); do
    shard_file=$(printf "train_%06d.npy" $i)
    if [ ! -f "$DATA_ROOT/$shard_file" ]; then
        echo "✗ Missing: $shard_file"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -eq 0 ]; then
    echo "✓ All 60 shards downloaded successfully!"
else
    echo "✗ $missing_files shards failed to download. Check the logs above."
fi