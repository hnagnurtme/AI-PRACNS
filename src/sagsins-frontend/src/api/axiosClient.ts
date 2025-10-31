import axios from 'axios';

const axiosClient = axios.create( {
    baseURL: import.meta.env.VITE_API_BASE_URL,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
} );

axiosClient.interceptors.request.use(
    ( config ) => {
        return config;
    },
    ( error ) => {
        return Promise.reject( error );
    }
);

axiosClient.interceptors.response.use(
    ( response ) => {
        return response;
    },
    ( error ) => {
        if ( error.response?.status === 401 ) {
            // Handle unauthorized access
            // localStorage.removeItem('token');
            // window.location.href = '/login';
        }
        return Promise.reject( error );
    }
);

export default axiosClient;