import React, { useEffect } from 'react';
import TestChart from '../components/packet/TestChart';
import { useNodes } from '../hooks/useNodes';

const Monitor: React.FC = () => {
    const { refetchNodes } = useNodes();

    useEffect( () => {
        refetchNodes().catch( ( error ) =>
            console.error( 'Failed to load Nodes data from API:', error )
        );
    }, [ refetchNodes ] );

    return (
        <div className="flex h-full w-full overflow-hidden bg-gray-100">
            <TestChart />
        </div>
    );
};

export default Monitor;