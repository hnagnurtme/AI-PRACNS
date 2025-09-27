import CesiumMap from '../components/CesiumMap';
import LoadingScreen from '../components/LoadingScreen';
import ErrorScreen from '../components/ErrorScreen';
import { useNodesApi } from '../hooks/useNodesApi';

export default function Home () {
    const { nodes, loading, error, retry } = useNodesApi();
    return (
        <div className="relative w-screen h-screen">
            <header className="absolute top-0 left-0 w-full z-20 bg-black/70 text-white py-2 px-4 flex items-center justify-between">
                <h1 className="text-lg font-bold tracking-wide">Satellite WebView</h1>
                <span className="text-sm">Powered by Cesium & React</span>
            </header>
            { loading && <LoadingScreen /> }
            { error && <ErrorScreen error={ error } onRetry={ retry } /> }
            { !loading && !error && <CesiumMap nodes={ nodes } /> }
        </div>
    );
}
