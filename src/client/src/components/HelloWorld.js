import React, { useState } from 'react';
import '../styles/HelloWorld.css';

const HelloWorld = () => {
  const [message, setMessage] = useState('Hello from AI-PRACNS Client!');
  const [counter, setCounter] = useState(0);

  const handleClick = () => {
    setCounter(counter + 1);
    setMessage(`Hello! You've clicked ${counter + 1} times.`);
  };

  return (
    <div className="hello-world">
      <h2>Hello World Component</h2>
      <div className="message-container">
        <p className="message">{message}</p>
        <button className="hello-button" onClick={handleClick}>
          Click me!
        </button>
      </div>
      <div className="info">
        <p>This is a sample component to test the client structure.</p>
        <p>Component features:</p>
        <ul>
          <li>React functional component with hooks</li>
          <li>State management with useState</li>
          <li>Event handling</li>
          <li>CSS styling</li>
        </ul>
      </div>
    </div>
  );
};

export default HelloWorld;