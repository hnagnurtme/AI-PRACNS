import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import HelloWorld from './components/HelloWorld';

test('renders Dashboard component', () => {
  render(
    <MemoryRouter>
      <Dashboard />
    </MemoryRouter>
  );
  const dashboardElement = screen.getByText(/Resource Allocation Dashboard/i);
  expect(dashboardElement).toBeInTheDocument();
});

test('renders system status in Dashboard', () => {
  render(
    <MemoryRouter>
      <Dashboard />
    </MemoryRouter>
  );
  const statusElement = screen.getByText(/System Status/i);
  expect(statusElement).toBeInTheDocument();
});

test('renders HelloWorld component with routing', () => {
  render(<HelloWorld />);
  const helloElement = screen.getByText(/Hello World Component/i);
  expect(helloElement).toBeInTheDocument();
});