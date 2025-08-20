import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './styles/App.css';
import HelloWorld from './components/HelloWorld';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>AI-PRACNS Client</h1>
          <p>AI-Powered Resource Allocation in Cloud and Network Systems</p>
        </header>
        <main className="App-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/hello" element={<HelloWorld />} />
          </Routes>
        </main>
        <footer className="App-footer">
          <p>&copy; 2024 AI-PRACNS Project</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;