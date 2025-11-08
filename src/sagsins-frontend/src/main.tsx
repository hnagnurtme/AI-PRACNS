import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { setupCesiumConfig } from './map/CesiumConfig'; 
import { WebSocketProvider } from './contexts/WebSocketContext';

// ************************************************
// 1. CẤU HÌNH TOÀN CỤC (Cesium và các thư viện khác)
// ************************************************

// Đường dẫn này phải trỏ đến thư mục chứa các file Workers, Assets, Widgets của Cesium
// (Thường được sao chép vào thư mục public/ Cesium khi build)
const CESIUM_BASE_URL = '/Cesium/'; 
setupCesiumConfig(CESIUM_BASE_URL);


// ************************************************
// 2. KHỞI TẠO ỨNG DỤNG REACT
// ************************************************

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <WebSocketProvider>
            <App />
        </WebSocketProvider>
    </React.StrictMode>,
);