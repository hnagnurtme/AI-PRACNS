import Layout from '../components/layout/Layout';
import { useNodesApi } from '../hooks/useNodesApi';
import '../index.css';

export default function Home() {
    const { nodes, loading, error, retry } = useNodesApi();
    
    return (
        <Layout 
            initialNodes={nodes}
            loading={loading}
            error={error}
            onRetry={retry}
        />
    );
}
