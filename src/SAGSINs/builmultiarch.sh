#!/bin/bash
# Build and push Docker multi-architecture images
echo "Setting up .jar file..."
mvn clean install 
if [ $? -ne 0 ]; then
  echo "Maven build failed. Exiting."
  exit 1
fi
echo "Setting up Docker buildx..."
echo "Building and pushing multi-architecture Docker image..."
docker buildx build --platform linux/amd64,linux/arm64 \
  -t trunganh0106/aipracns \
  --push .

echo "Done!"