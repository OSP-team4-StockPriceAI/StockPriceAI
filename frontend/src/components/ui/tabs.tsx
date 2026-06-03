"use client";

import { type ReactNode } from "react";

interface Tab {
  id: string;
  label: string;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (id: string) => void;
  children: ReactNode;
}

interface TabPanelProps {
  id: string;
  activeTab: string;
  children: ReactNode;
}

export function Tabs({ tabs, activeTab, onChange, children }: TabsProps) {
  return (
    <div>
      <div className="flex gap-1 border-b border-[#2d3748] overflow-x-auto overflow-y-hidden">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={[
              "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
              "border-b-2 -mb-px focus:outline-none",
              activeTab === tab.id
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-[#718096] hover:text-[#a0aec0]",
            ].join(" ")}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

export function TabPanel({ id, activeTab, children }: TabPanelProps) {
  if (id !== activeTab) return null;
  return <div>{children}</div>;
}
