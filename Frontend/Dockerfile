FROM node:20-alpine

WORKDIR /app

# Copy package files first
COPY package*.json ./
COPY yarn.lock* ./

# Copy scripts directory needed for postinstall
COPY scripts/ ./scripts/

# Install dependencies (this will run postinstall script)
RUN npm install

# Copy rest of application code
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]

