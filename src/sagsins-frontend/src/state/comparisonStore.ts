// src/state/comparisonStore.ts
import { create } from "zustand";
import type { ComparisonData as PacketComparison } from "../types/ComparisonTypes";

interface ComparisonState {
    comparison: PacketComparison | null;
    setComparison: (data: PacketComparison) => void;
}

export const useComparisonStore = create<ComparisonState>((set) => ({
    comparison: null,
    setComparison: (data) => set({ comparison: data }),
}));
