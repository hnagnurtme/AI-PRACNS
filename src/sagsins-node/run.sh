#!/bin/bash
# run.sh - tự động load env, build và chạy project Maven

# 1️⃣ Load biến môi trường từ .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅ Loaded environment variables from .env"
else
  echo "⚠️ .env file not found"
fi

# 2️⃣ Clean và compile Maven project
echo "🔨 Running mvn clean compile..."
mvn clean compile
if [ $? -ne 0 ]; then
  echo "❌ Maven build failed"
  exit 1
fi

# 3️⃣ Run main class
echo "🚀 Running SimulationMain..."
mvn exec:java -Dexec.mainClass="com.sagin.util.SimulationMain"
