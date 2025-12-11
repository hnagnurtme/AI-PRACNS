// MongoDB initialization script
// This script runs automatically when MongoDB container starts for the first time

// Switch to the aiprancs database
db = db.getSiblingDB('aiprancs');

// Create collections with validation (optional)
// Collections will be created automatically when data is inserted

// Create indexes for better performance
// These will be created by the backend application

print('✅ MongoDB initialization completed');

