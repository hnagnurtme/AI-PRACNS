import React, { useEffect } from 'react';
import CesiumViewer from '../map/CesiumViewer';
import Sidebar from '../components/Sidebar';
import NodeDetailCard from '../components/nodes/NodeDetailCard';
import { useNodeStore } from '../state/nodeStore';
import { useNodes } from '../hooks/useNodes';

const Dashboard: React.FC = () => {
    const { nodes, selectedNode } = useNodeStore();
    const { refetchNodes } = useNodes();

    useEffect( () => {
        refetchNodes().catch( ( error ) =>
            console.error( 'Failed to load Nodes data from API:', error )
        );
    }, [ refetchNodes ] );

    return (
        <div className="flex w-screen h-screen overflow-hidden bg-gray-100">
            {/* Sidebar danh sách node */ }
            <Sidebar nodes={ nodes } />

            {/* Khu vực bản đồ (relative là đúng rồi) */ }
            <div className="relative flex-grow">
                <CesiumViewer nodes={ nodes } />

                { selectedNode && (
                    <NodeDetailCard node={ selectedNode } />
                ) }
            </div>
        </div>
    );
};

export default Dashboard;