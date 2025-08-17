#!/bin/bash

# Simple demo script

echo "Hello, this is a demo shell script!"

# Variables demonstration
name="User"
current_date=$(date +"%Y-%m-%d")

echo "Hello, $name!"
echo "Today's date is: $current_date"

# Simple loop demonstration
echo -e "\nCounting from 1 to 5:"
for i in {1..5}; do
    echo "Number $i"
    sleep 0.5
done

# Conditional demonstration
echo -e "\nRandom number demonstration:"
random_number=$((RANDOM % 10 + 1))
echo "Generated random number: $random_number"

if [ $random_number -gt 5 ]; then
    echo "The number is greater than 5!"
else
    echo "The number is 5 or less!"
fi

echo -e "\nDemo completed successfully!"