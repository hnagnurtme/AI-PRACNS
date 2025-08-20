# AI-PRACNS Client

Frontend client for AI-Powered Resource Allocation in Cloud and Network Systems project.

## Overview

This React-based client application provides a user interface for interacting with the resource allocation system. It allows users to monitor system status, view resource nodes, and manage allocation requests.

## Project Structure

```
src/client/
├── public/                 # Public assets
│   ├── index.html         # Main HTML template
│   └── manifest.json      # PWA manifest
├── src/                   # Source code
│   ├── components/        # React components
│   │   └── HelloWorld.js  # Sample component
│   ├── pages/            # Page components
│   │   └── Dashboard.js  # Main dashboard
│   ├── styles/           # CSS stylesheets
│   │   ├── index.css     # Global styles
│   │   ├── App.css       # App component styles
│   │   ├── HelloWorld.css
│   │   └── Dashboard.css
│   ├── utils/            # Utility functions
│   │   ├── api.js        # API communication
│   │   ├── config.js     # Configuration
│   │   └── helpers.js    # Helper functions
│   ├── App.js            # Main App component
│   └── index.js          # Entry point
├── package.json          # Dependencies and scripts
├── .env.example          # Environment variables template
├── .eslintrc.json        # ESLint configuration
├── .prettierrc           # Prettier configuration
└── .gitignore           # Git ignore rules
```

## Features

- **Dashboard**: System status overview and resource monitoring
- **Component-based architecture**: Modular React components
- **Responsive design**: Mobile-friendly interface
- **API integration**: Communication with backend servers
- **Code quality**: ESLint and Prettier configuration
- **Environment configuration**: Flexible environment setup

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn package manager

### Installation

1. Navigate to the client directory:
```bash
cd src/client
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Configure environment variables in `.env` file:
```bash
REACT_APP_API_BASE_URL=http://localhost:8080/api
REACT_APP_AI_SERVER_URL=http://localhost:8081/api
```

### Development

Start the development server:
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`.

### Building for Production

Create an optimized production build:
```bash
npm run build
```

The build files will be generated in the `build/` directory.

### Code Quality

Run ESLint to check code quality:
```bash
npm run lint
```

Fix ESLint issues automatically:
```bash
npm run lint:fix
```

Format code with Prettier:
```bash
npm run format
```

### Testing

Run tests:
```bash
npm test
```

## API Integration

The client communicates with two main servers:

1. **Resource Allocation Server** (`localhost:8080`):
   - System status monitoring
   - Node management
   - Request handling

2. **AI Server** (`localhost:8081`):
   - Optimization algorithms
   - Performance prediction
   - Strategy management

## Component Guide

### HelloWorld Component
A sample component demonstrating:
- React hooks (useState)
- Event handling
- CSS styling
- Component structure

### Dashboard Component
The main interface showing:
- System status
- Active resource nodes
- Quick actions

## Environment Variables

Create a `.env` file based on `.env.example`:

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_API_BASE_URL` | Resource allocation server URL | `http://localhost:8080/api` |
| `REACT_APP_AI_SERVER_URL` | AI server URL | `http://localhost:8081/api` |
| `REACT_APP_NAME` | Application name | `AI-PRACNS Client` |
| `REACT_APP_VERSION` | Application version | `1.0.0` |
| `REACT_APP_DEBUG` | Enable debug mode | `false` |

## Technologies Used

- **React** (v18): Frontend framework
- **React Router**: Client-side routing
- **Axios**: HTTP client for API requests
- **ESLint**: Code linting
- **Prettier**: Code formatting
- **CSS3**: Styling and animations

## Development Guidelines

1. **Components**: Create reusable components in `src/components/`
2. **Pages**: Add new pages in `src/pages/`
3. **Styles**: Keep component-specific styles in `src/styles/`
4. **Utils**: Add utility functions in `src/utils/`
5. **API**: Use the `api.js` helper for server communication

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Follow the existing code style
2. Run linting and formatting before committing
3. Add tests for new features
4. Update documentation as needed

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `package.json` or stop other services
2. **API connection errors**: Check server URLs in `.env` file
3. **Build failures**: Clear `node_modules` and reinstall dependencies

### Getting Help

- Check the console for error messages
- Verify environment variables are set correctly
- Ensure backend servers are running
- Review the network tab in browser dev tools

## License

This project is part of the AI-PRACNS system. See the main project license for details.