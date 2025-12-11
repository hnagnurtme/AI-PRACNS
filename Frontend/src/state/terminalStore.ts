import { create } from 'zustand';
import type { UserTerminal } from '../types/UserTerminalTypes';

interface TerminalState {
    terminals: UserTerminal[];
    selectedTerminal: UserTerminal | null;
    sourceTerminal: UserTerminal | null; // Terminal được chọn làm source
    destinationTerminal: UserTerminal | null; // Terminal được chọn làm destination
    
    setTerminals: (terminals: UserTerminal[]) => void;
    setSelectedTerminal: (terminal: UserTerminal | null) => void;
    setSourceTerminal: (terminal: UserTerminal | null) => void;
    setDestinationTerminal: (terminal: UserTerminal | null) => void;
    clearPacketSelection: () => void; // Clear cả source và destination
    updateTerminalInStore: (updatedTerminal: UserTerminal) => void;
    removeTerminalFromStore: (terminalId: string) => void;
}

export const useTerminalStore = create<TerminalState>((set) => ({
    terminals: [],
    selectedTerminal: null,
    sourceTerminal: null,
    destinationTerminal: null,
    
    setTerminals: (terminals) => set({ terminals }),
    
    setSelectedTerminal: (terminal) => set({ selectedTerminal: terminal }),
    
    setSourceTerminal: (terminal) => set({ sourceTerminal: terminal }),
    
    setDestinationTerminal: (terminal) => set({ destinationTerminal: terminal }),
    
    clearPacketSelection: () => set({ sourceTerminal: null, destinationTerminal: null }),

    updateTerminalInStore: (updatedTerminal) => 
        set((state) => ({
            terminals: state.terminals.map(terminal => 
                terminal.terminalId === updatedTerminal.terminalId ? updatedTerminal : terminal
            ),
            selectedTerminal: state.selectedTerminal && state.selectedTerminal.terminalId === updatedTerminal.terminalId 
                ? updatedTerminal : state.selectedTerminal,
            sourceTerminal: state.sourceTerminal && state.sourceTerminal.terminalId === updatedTerminal.terminalId 
                ? updatedTerminal : state.sourceTerminal,
            destinationTerminal: state.destinationTerminal && state.destinationTerminal.terminalId === updatedTerminal.terminalId 
                ? updatedTerminal : state.destinationTerminal,
        })),

    removeTerminalFromStore: (terminalId) => 
        set((state) => ({
            terminals: state.terminals.filter(terminal => terminal.terminalId !== terminalId),
            selectedTerminal: state.selectedTerminal && state.selectedTerminal.terminalId === terminalId 
                ? null : state.selectedTerminal,
            sourceTerminal: state.sourceTerminal && state.sourceTerminal.terminalId === terminalId 
                ? null : state.sourceTerminal,
            destinationTerminal: state.destinationTerminal && state.destinationTerminal.terminalId === terminalId 
                ? null : state.destinationTerminal,
        }))
}));

