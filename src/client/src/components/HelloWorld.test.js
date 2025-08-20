import { render, screen, fireEvent } from '@testing-library/react';
import HelloWorld from './HelloWorld';

test('renders HelloWorld component', () => {
  render(<HelloWorld />);
  const titleElement = screen.getByText(/Hello World Component/i);
  expect(titleElement).toBeInTheDocument();
});

test('displays initial message', () => {
  render(<HelloWorld />);
  const messageElement = screen.getByText(/Hello from AI-PRACNS Client!/i);
  expect(messageElement).toBeInTheDocument();
});

test('updates message when button is clicked', () => {
  render(<HelloWorld />);
  const button = screen.getByText(/Click me!/i);
  
  fireEvent.click(button);
  
  const updatedMessage = screen.getByText(/Hello! You've clicked 1 times./i);
  expect(updatedMessage).toBeInTheDocument();
});

test('increments counter on multiple clicks', () => {
  render(<HelloWorld />);
  const button = screen.getByText(/Click me!/i);
  
  fireEvent.click(button);
  fireEvent.click(button);
  fireEvent.click(button);
  
  const updatedMessage = screen.getByText(/Hello! You've clicked 3 times./i);
  expect(updatedMessage).toBeInTheDocument();
});