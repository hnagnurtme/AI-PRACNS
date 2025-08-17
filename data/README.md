# Data Directory

This directory contains all the data required for the operation of the project.

## Structure

- **input_data/**: Raw input data files
- **request_history/**: Logs of past requests and their responses
- **node_states/**: Saved states of processing nodes

## Usage

Data in this directory is accessed by the main application to:
1. Process input requests
2. Track history for analytics purposes
3. Maintain state between system restarts

## Maintenance

Regular backups of this directory are recommended to prevent data loss.

Note: Large binary files should not be committed to version control.