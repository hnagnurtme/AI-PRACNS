import React, { useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { NodeAnalysis } from '../../types/NodeAnalysisTypes';
import type { NodeDTO } from '../../types/NodeTypes';

interface NetworkGraphProps {
    analysis: NodeAnalysis | null;
    allNodes: NodeDTO[];
    loading?: boolean;
}

const NetworkGraph: React.FC<NetworkGraphProps> = ({ analysis, allNodes, loading }) => {
    // Build graph data from analysis
    const graphData = useMemo(() => {
        if (!analysis) return { nodes: [], links: [] };

        const nodeMap = new Map<string, NodeDTO>();
        allNodes.forEach(node => {
            nodeMap.set(node.nodeId, node);
        });

        // Center node (selected node)
        const centerNode = nodeMap.get(analysis.nodeId);
        if (!centerNode) return { nodes: [], links: [] };

        const nodes: any[] = [
            {
                id: analysis.nodeId,
                name: analysis.nodeName,
                type: 'center',
                node: centerNode,
            }
        ];

        const links: any[] = [];

        // Add best links (neighbors)
        analysis.bestLinks.forEach((link) => {
            const neighborNode = nodeMap.get(link.nodeId);
            if (neighborNode) {
                nodes.push({
                    id: link.nodeId,
                    name: link.nodeName,
                    type: 'neighbor',
                    quality: link.quality,
                    score: link.score,
                    node: neighborNode,
                });

                links.push({
                    source: analysis.nodeId,
                    target: link.nodeId,
                    type: 'best-link',
                    quality: link.quality,
                    latency: link.latency,
                    bandwidth: link.bandwidth,
                    distance: link.distance,
                });
            }
        });

        // Add upcoming satellites
        analysis.upcomingSatellites.forEach((sat) => {
            if (!nodes.find(n => n.id === sat.nodeId)) {
                nodes.push({
                    id: sat.nodeId,
                    name: sat.nodeName,
                    type: 'upcoming',
                    willBeInRange: sat.willBeInRange,
                    estimatedArrivalIn: sat.estimatedArrivalIn,
                    node: nodeMap.get(sat.nodeId),
                });

                links.push({
                    source: analysis.nodeId,
                    target: sat.nodeId,
                    type: 'upcoming',
                    dashed: true,
                });
            }
        });

        // Add degrading nodes
        analysis.degradingNodes.forEach((degrading) => {
            const degradingNode = nodeMap.get(degrading.nodeId);
            if (degradingNode && !nodes.find(n => n.id === degrading.nodeId)) {
                nodes.push({
                    id: degrading.nodeId,
                    name: degrading.nodeName,
                    type: 'degrading',
                    severity: degrading.severity,
                    predictedDegradationIn: degrading.predictedDegradationIn,
                    node: degradingNode,
                });

                links.push({
                    source: analysis.nodeId,
                    target: degrading.nodeId,
                    type: 'degrading',
                    severity: degrading.severity,
                });
            }
        });

        return { nodes, links };
    }, [analysis, allNodes]);

    const getNodeColor = (node: any) => {
        if (node.type === 'center') return '#3b82f6'; // Blue
        if (node.type === 'neighbor') {
            switch (node.quality) {
                case 'excellent': return '#10b981'; // Green
                case 'good': return '#84cc16'; // Lime
                case 'fair': return '#f59e0b'; // Amber
                case 'poor': return '#ef4444'; // Red
                default: return '#6b7280'; // Gray
            }
        }
        if (node.type === 'upcoming') return '#8b5cf6'; // Purple
        if (node.type === 'degrading') {
            switch (node.severity) {
                case 'critical': return '#dc2626'; // Red
                case 'warning': return '#f59e0b'; // Amber
                case 'minor': return '#eab308'; // Yellow
                default: return '#6b7280';
            }
        }
        return '#6b7280';
    };

    const getLinkColor = (link: any) => {
        if (link.type === 'best-link') {
            switch (link.quality) {
                case 'excellent': return '#10b981';
                case 'good': return '#84cc16';
                case 'fair': return '#f59e0b';
                case 'poor': return '#ef4444';
                default: return '#6b7280';
            }
        }
        if (link.type === 'upcoming') return '#8b5cf6';
        if (link.type === 'degrading') {
            switch (link.severity) {
                case 'critical': return '#dc2626';
                case 'warning': return '#f59e0b';
                case 'minor': return '#eab308';
                default: return '#6b7280';
            }
        }
        return '#6b7280';
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full bg-gray-50 rounded-lg">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading network analysis...</p>
                </div>
            </div>
        );
    }

    if (!analysis || graphData.nodes.length === 0) {
        return (
            <div className="flex items-center justify-center h-full bg-gray-50 rounded-lg">
                <p className="text-gray-600">Select a node to view network graph</p>
            </div>
        );
    }

    return (
        <div className="w-full h-full bg-white rounded-lg shadow-lg border border-gray-200">
            <div className="p-4 border-b border-gray-200">
                <h3 className="text-lg font-bold text-gray-800">Network Graph: {analysis.nodeName}</h3>
                <p className="text-sm text-gray-600 mt-1">
                    Best Links: {analysis.bestLinks.length} | 
                    Upcoming: {analysis.upcomingSatellites.length} | 
                    Degrading: {analysis.degradingNodes.length}
                </p>
            </div>
            <div className="w-full h-[600px]">
                <ForceGraph2D
                    graphData={graphData}
                    nodeLabel={(node: any) => {
                        let label = `${node.name}\n`;
                        if (node.type === 'neighbor') {
                            label += `Quality: ${node.quality}\nScore: ${node.score.toFixed(1)}`;
                        } else if (node.type === 'upcoming') {
                            label += `Arrives in: ${Math.floor(node.estimatedArrivalIn / 60)}m`;
                        } else if (node.type === 'degrading') {
                            label += `Degrades in: ${Math.floor(node.predictedDegradationIn / 60)}m\nSeverity: ${node.severity}`;
                        }
                        return label;
                    }}
                    nodeColor={(node: any) => getNodeColor(node)}
                    nodeVal={(node: any) => {
                        if (node.type === 'center') return 15;
                        if (node.type === 'neighbor') return 10;
                        return 8;
                    }}
                    linkLabel={(link: any) => {
                        if (link.type === 'best-link') {
                            return `Latency: ${link.latency.toFixed(1)}ms\nBandwidth: ${link.bandwidth.toFixed(1)}Mbps\nDistance: ${link.distance.toFixed(1)}km`;
                        }
                        if (link.type === 'upcoming') {
                            return 'Upcoming satellite';
                        }
                        if (link.type === 'degrading') {
                            return `Degrading node (${link.severity})`;
                        }
                        return '';
                    }}
                    linkColor={(link: any) => getLinkColor(link)}
                    linkWidth={(link: any) => {
                        if (link.type === 'best-link') {
                            return link.quality === 'excellent' ? 3 : link.quality === 'good' ? 2.5 : 2;
                        }
                        return 1.5;
                    }}
                    linkDirectionalArrowLength={6}
                    linkDirectionalArrowRelPos={1}
                    linkCurvature={0.1}
                    onNodeHover={(_node: any) => {
                        // Optional: show tooltip
                    }}
                    cooldownTicks={100}
                    onEngineStop={() => {
                        // Graph layout stabilized
                    }}
                />
            </div>
            <div className="p-4 border-t border-gray-200 bg-gray-50">
                <div className="grid grid-cols-4 gap-4 text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                        <span>Selected Node</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        <span>Best Links</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                        <span>Upcoming</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                        <span>Degrading</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default NetworkGraph;

