import React from 'react';
import Dashboard from './pages/Dashboard'; 

/**
 * Thành phần chính của ứng dụng.
 * Nếu ứng dụng có nhiều trang, đây sẽ là nơi cấu hình React Router.
 */
const App: React.FC = () => {
    return (
        // Dashboard chứa tất cả UI và Map logic
        <Dashboard /> 
    );
};

export default App;