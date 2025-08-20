import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Dashboard.css';

const Dashboard = () => {
  const [systemStatus, setSystemStatus] = useState('Checking...');
  const [resourceNodes, setResourceNodes] = useState([]);

  useEffect(() => {
    // Simulate system status check
    setTimeout(() => {
      setSystemStatus('Online');
      setResourceNodes([
        { id: 'SAT-001', type: 'Satellite', status: 'Active', load: '45%' },
        { id: 'UAV-002', type: 'UAV', status: 'Active', load: '23%' },
        { id: 'GS-003', type: 'Ground Station', status: 'Active', load: '67%' },
      ]);
    }, 1000);
  }, []);

  return (
    <div className="dashboard">
      <h2>Resource Allocation Dashboard</h2>
      
      <div className="dashboard-grid">
        <div className="status-card">
          <h3>System Status</h3>
          <p className={`status ${systemStatus.toLowerCase()}`}>
            {systemStatus}
          </p>
        </div>

        <div className="nodes-card">
          <h3>Active Nodes</h3>
          <div className="nodes-list">
            {resourceNodes.map(node => (
              <div key={node.id} className="node-item">
                <span className="node-id">{node.id}</span>
                <span className="node-type">{node.type}</span>
                <span className="node-status">{node.status}</span>
                <span className="node-load">{node.load}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="actions-card">
          <h3>Quick Actions</h3>
          <div className="action-buttons">
            <Link to="/hello" className="action-link">
              Test Component
            </Link>
            <button className="action-button" disabled>
              Request Allocation
            </button>
            <button className="action-button" disabled>
              View History
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;