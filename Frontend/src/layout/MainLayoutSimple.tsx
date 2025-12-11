import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface MainLayoutSimpleProps {
  children: React.ReactNode;
}

const MainLayoutSimple: React.FC<MainLayoutSimpleProps> = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Bản đồ 3D' },
    { path: '/cesium-test', label: 'Cesium Test' },
    { path: '/simulation', label: 'Mô phỏng' },
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/test', label: 'Test' },
  ];

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* Navigation Bar */}
      <nav style={{
        background: 'linear-gradient(to right, #4f46e5, #9333ea)',
        color: 'white',
        padding: '0 20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <div style={{
          maxWidth: '1280px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: '64px'
        }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>AIPRANCS</h1>
          </div>
          <div style={{ display: 'flex', gap: '4px' }}>
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 16px',
                    borderRadius: '8px',
                    textDecoration: 'none',
                    color: 'white',
                    backgroundColor: isActive ? 'rgba(255,255,255,0.2)' : 'transparent',
                    fontWeight: isActive ? 'bold' : 'normal',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main>{children}</main>
    </div>
  );
};

export default MainLayoutSimple;

